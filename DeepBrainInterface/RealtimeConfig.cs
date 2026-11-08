using Bonsai;
using System;
using System.ComponentModel;
using System.Diagnostics;
using System.Reactive.Disposables;
using System.Reactive.Linq;
using System.Runtime;
using System.Runtime.InteropServices;

namespace DeepBrainInterface
{
    /// <summary>Snapshot of the settings RealtimeConfig actually managed to apply.</summary>
    public struct RealtimeConfigState
    {
        public ProcessPriorityClass ProcessPriority { get; set; }
        public bool TimerResolution1ms { get; set; }
        public double TimerResolutionMs { get; set; }   // actual system timer resolution after the request (0 = not set)
        public GCLatencyMode GcLatencyMode { get; set; }
        public bool IsServerGC { get; set; }
        public bool PowerThrottlingDisabled { get; set; }
        public long ProcessorAffinityMask { get; set; }
        public long PressureBallastBytes { get; set; }   // GC memory-pressure ballast outstanding in this process (0 = none)
    }

    [Description("One-shot real-time configuration of the Bonsai process: priority class, 1 ms timer resolution, " +
                 "GC latency mode, optional GC memory-pressure ballast, power-throttling (EcoQoS) opt-out and optional P-core affinity. " +
                 "Emits the applied state once at workflow start and reverts everything when the workflow stops. " +
                 "Use instead of the priority side effects inside detector nodes (set their ManagePriority to false).")]
    public class RealtimeConfig : Source<RealtimeConfigState>
    {
        // --- Win32 ---------------------------------------------------------------------------
        const uint ProcessPowerThrottling = 4;                               // PROCESS_INFORMATION_CLASS
        const uint PROCESS_POWER_THROTTLING_CURRENT_VERSION = 1;
        const uint PROCESS_POWER_THROTTLING_EXECUTION_SPEED = 0x1;           // EcoQoS
        const uint PROCESS_POWER_THROTTLING_IGNORE_TIMER_RESOLUTION = 0x4;   // Win11 background timer coalescing
        const int RelationProcessorCore = 0;

        [StructLayout(LayoutKind.Sequential)]
        struct PROCESS_POWER_THROTTLING_STATE
        {
            public uint Version;
            public uint ControlMask;
            public uint StateMask;
        }

        [DllImport("winmm.dll", EntryPoint = "timeBeginPeriod")] static extern uint TimeBeginPeriod(uint uMilliseconds);
        [DllImport("winmm.dll", EntryPoint = "timeEndPeriod")] static extern uint TimeEndPeriod(uint uMilliseconds);
        [DllImport("ntdll.dll")] static extern int NtSetTimerResolution(uint DesiredResolution, bool SetResolution, out uint CurrentResolution);
        [DllImport("ntdll.dll")] static extern int NtQueryTimerResolution(out uint MinimumResolution, out uint MaximumResolution, out uint CurrentResolution);
        [DllImport("kernel32.dll")] static extern IntPtr GetCurrentProcess();
        [DllImport("kernel32.dll", SetLastError = true)]
        static extern bool SetProcessInformation(IntPtr hProcess, uint ProcessInformationClass,
                                                 ref PROCESS_POWER_THROTTLING_STATE ProcessInformation, uint ProcessInformationSize);
        [DllImport("kernel32.dll", SetLastError = true)]
        static extern bool GetLogicalProcessorInformationEx(int RelationshipType, IntPtr Buffer, ref uint ReturnedLength);

        // --- Properties ----------------------------------------------------------------------
        [Description("Process priority class. High is the recommended default; RealTime needs administrator rights " +
                     "and starves driver, disk and GC threads (silently falls back to High without admin).")]
        public ProcessPriorityClass ProcessPriority { get; set; } = ProcessPriorityClass.RealTime;

        [Description("Requested system timer resolution in ms (NtSetTimerResolution). 0.5 is the hardware minimum on most boards; " +
                     "1 = timeBeginPeriod(1); 0 = leave the system default (15.6 ms). The request is made from the Bonsai process, " +
                     "so no GlobalTimerResolutionRequests registry key is needed. The achieved value is reported in TimerResolutionMs.")]
        public double TimerResolutionMs { get; set; } = 0.5;

        [Browsable(false)] // legacy; kept so old workflow files load. TimerResolutionMs replaces it.
        public bool TimerResolution1ms { get; set; } = true;

        [Description("GC latency mode. SustainedLowLatency suppresses blocking gen2 collections (needs concurrent GC, Bonsai default).")]
        public GCLatencyMode GcLatencyMode { get; set; } = GCLatencyMode.Interactive;   // Bonsai Mats are finalizable: SustainedLowLatency lets gen2 grow until a blocking collection freezes the process

        [Description("Opt the process out of power throttling (EcoQoS / E-core steering when minimized) and, on Windows 11, " +
                     "out of background timer-resolution coalescing.")]
        public bool DisablePowerThrottling { get; set; } = true;

        [Description("Restrict the whole process to the performance (P) cores of a hybrid CPU. Process-wide mask, not a pin: " +
                     "threads may still migrate between P-cores. Off by default; measure before enabling.")]
        public bool RestrictToPerformanceCores { get; set; } = false;

        [Description("Explicit process affinity mask (hex, e.g. 3C00 = logical CPUs 10-13). Overrides RestrictToPerformanceCores " +
                     "when non-empty, so cores 0/1 can be left to the system. Empty = no explicit mask.")]
        public string AffinityMaskHex { get; set; } = "";

        [Description("GC memory-pressure ballast in bytes (0 = off). Applied once per process with GC.AddMemoryPressure and never removed. " +
                     "On .NET Framework the pressure threshold is reset after every pressure-triggered collection and shrunk by 5% on every " +
                     "RemoveMemoryPressure, so a driver that registers pressure per frame (clroni, tens of thousands of frames/s) forces " +
                     "~100 induced collections per second whatever the byte count. A large ballast (8589934592 = 8 GB) parks the threshold " +
                     "~5% above the outstanding total, out of reach of per-frame add/remove pairs. Costs one induced GC at start and no memory. " +
                     "Verify with an ETW GC trace: GC/Start reason InducedNoForce should fall to ~0/s. The amount outstanding is reported in " +
                     "PressureBallastBytes; raising the value in the same process adds only the difference.")]
        public long PressureBallastBytes { get; set; } = 0;

        static readonly object ballastLock = new object();
        static long ballastApplied;                      // process-wide, so an editor restart of the workflow does not add it twice

        public override IObservable<RealtimeConfigState> Generate()
        {
            return Observable.Create<RealtimeConfigState>(observer =>
            {
                var undo = new CompositeDisposable();
                var state = new RealtimeConfigState();
                var proc = Process.GetCurrentProcess();

                // 1. Process priority class
                try
                {
                    var previous = proc.PriorityClass;
                    proc.PriorityClass = ProcessPriority;
                    proc.Refresh();
                    undo.Add(Disposable.Create(() => { try { proc.PriorityClass = previous; } catch { } }));
                }
                catch { }
                try { state.ProcessPriority = proc.PriorityClass; } catch { }

                // 2. Timer resolution
                if (TimerResolutionMs > 0)
                {
                    try
                    {
                        uint desired = (uint)Math.Round(TimerResolutionMs * 10000.0);   // 100 ns units
                        uint actual;
                        if (NtSetTimerResolution(desired, true, out actual) == 0)
                        {
                            state.TimerResolution1ms = actual <= 10000;
                            state.TimerResolutionMs = actual / 10000.0;
                            undo.Add(Disposable.Create(() => { uint dummy; try { NtSetTimerResolution(desired, false, out dummy); } catch { } }));
                        }
                        else if (TimeBeginPeriod(1) == 0)   // fallback: 1 ms via winmm
                        {
                            state.TimerResolution1ms = true;
                            state.TimerResolutionMs = 1.0;
                            undo.Add(Disposable.Create(() => { try { TimeEndPeriod(1); } catch { } }));
                        }
                    }
                    catch { }
                }

                // 3. GC latency mode
                try
                {
                    var previous = GCSettings.LatencyMode;
                    GCSettings.LatencyMode = GcLatencyMode;
                    undo.Add(Disposable.Create(() => { try { GCSettings.LatencyMode = previous; } catch { } }));
                }
                catch { }
                state.GcLatencyMode = GCSettings.LatencyMode;
                state.IsServerGC = GCSettings.IsServerGC;

                // 3b. GC memory-pressure ballast: monotone per process, deliberately not undone on stop
                //     (RemoveMemoryPressure of >= threshold/4 resets the threshold to 100 KB and the per-frame storm returns).
                if (PressureBallastBytes > 0)
                {
                    try
                    {
                        lock (ballastLock)
                        {
                            long delta = PressureBallastBytes - ballastApplied;
                            if (delta > 0)
                            {
                                GC.AddMemoryPressure(delta);
                                ballastApplied += delta;
                            }
                        }
                    }
                    catch { }
                }
                state.PressureBallastBytes = ballastApplied;

                // 4. Power throttling / EcoQoS opt-out
                if (DisablePowerThrottling)
                {
                    try
                    {
                        var size = (uint)Marshal.SizeOf(typeof(PROCESS_POWER_THROTTLING_STATE));
                        var disable = new PROCESS_POWER_THROTTLING_STATE
                        {
                            Version = PROCESS_POWER_THROTTLING_CURRENT_VERSION,
                            ControlMask = PROCESS_POWER_THROTTLING_EXECUTION_SPEED | PROCESS_POWER_THROTTLING_IGNORE_TIMER_RESOLUTION,
                            StateMask = 0 // flag in ControlMask but not in StateMask = feature OFF (no throttling, honour timer requests)
                        };
                        if (SetProcessInformation(GetCurrentProcess(), ProcessPowerThrottling, ref disable, size))
                        {
                            state.PowerThrottlingDisabled = true;
                            undo.Add(Disposable.Create(() =>
                            {
                                var systemManaged = new PROCESS_POWER_THROTTLING_STATE
                                {
                                    Version = PROCESS_POWER_THROTTLING_CURRENT_VERSION,
                                    ControlMask = 0,
                                    StateMask = 0
                                };
                                try { SetProcessInformation(GetCurrentProcess(), ProcessPowerThrottling, ref systemManaged, size); } catch { }
                            }));
                        }
                    }
                    catch { }
                }

                // 5. Optional process-wide P-core affinity
                try
                {
                    long explicitMask = 0;
                    if (!string.IsNullOrWhiteSpace(AffinityMaskHex))
                        long.TryParse(AffinityMaskHex.Trim().Replace("0x", "").Replace("0X", ""),
                                      System.Globalization.NumberStyles.HexNumber, null, out explicitMask);
                    if (explicitMask != 0 || RestrictToPerformanceCores)
                    {
                        long mask = explicitMask != 0 ? explicitMask : FindPerformanceCoreMask();
                        if (mask != 0)
                        {
                            var previous = proc.ProcessorAffinity;
                            proc.ProcessorAffinity = (IntPtr)mask;
                            undo.Add(Disposable.Create(() => { try { proc.ProcessorAffinity = previous; } catch { } }));
                        }
                    }
                    state.ProcessorAffinityMask = (long)proc.ProcessorAffinity;
                }
                catch { }

                undo.Add(proc); // disposed last (CompositeDisposable disposes in insertion order)
                observer.OnNext(state);
                return undo;
            });
        }

        /// <summary>
        /// Affinity mask of every core in the highest EfficiencyClass (the P-cores on a hybrid CPU).
        /// Returns 0 if the information is unavailable (then nothing is changed).
        /// </summary>
        static long FindPerformanceCoreMask()
        {
            uint len = 0;
            GetLogicalProcessorInformationEx(RelationProcessorCore, IntPtr.Zero, ref len);
            if (len == 0) return 0;
            IntPtr buf = Marshal.AllocHGlobal((int)len);
            try
            {
                if (!GetLogicalProcessorInformationEx(RelationProcessorCore, buf, ref len)) return 0;
                int bestClass = -1;
                long bestMask = 0;
                int offset = 0;
                while (offset + 8 <= (int)len)
                {
                    int relationship = Marshal.ReadInt32(buf, offset + 0);
                    int size = Marshal.ReadInt32(buf, offset + 4);
                    if (size <= 0) break;
                    if (relationship == RelationProcessorCore)
                    {
                        // PROCESSOR_RELATIONSHIP: Flags@8, EfficiencyClass@9, Reserved[20], GroupCount@30, GroupMask[0].Mask@32 (x64)
                        int effClass = Marshal.ReadByte(buf, offset + 9);
                        long mask = (long)Marshal.ReadIntPtr(buf, offset + 32);
                        if (effClass > bestClass) { bestClass = effClass; bestMask = mask; }
                        else if (effClass == bestClass) bestMask |= mask;
                    }
                    offset += size;
                }
                return bestMask;
            }
            finally { Marshal.FreeHGlobal(buf); }
        }
    }
}
