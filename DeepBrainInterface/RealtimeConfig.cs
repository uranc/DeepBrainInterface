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
        public GCLatencyMode GcLatencyMode { get; set; }
        public bool IsServerGC { get; set; }
        public bool PowerThrottlingDisabled { get; set; }
        public long ProcessorAffinityMask { get; set; }
    }

    [Description("One-shot real-time configuration of the Bonsai process: priority class, 1 ms timer resolution, " +
                 "GC latency mode, power-throttling (EcoQoS) opt-out and optional P-core affinity. " +
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
        [DllImport("kernel32.dll")] static extern IntPtr GetCurrentProcess();
        [DllImport("kernel32.dll", SetLastError = true)]
        static extern bool SetProcessInformation(IntPtr hProcess, uint ProcessInformationClass,
                                                 ref PROCESS_POWER_THROTTLING_STATE ProcessInformation, uint ProcessInformationSize);
        [DllImport("kernel32.dll", SetLastError = true)]
        static extern bool GetLogicalProcessorInformationEx(int RelationshipType, IntPtr Buffer, ref uint ReturnedLength);

        // --- Properties ----------------------------------------------------------------------
        [Description("Process priority class. High is the recommended default; RealTime needs administrator rights " +
                     "and starves driver, disk and GC threads (silently falls back to High without admin).")]
        public ProcessPriorityClass ProcessPriority { get; set; } = ProcessPriorityClass.High;

        [Description("Call timeBeginPeriod(1) so timer-based waits wake on a 1 ms grid instead of 15.6 ms.")]
        public bool TimerResolution1ms { get; set; } = true;

        [Description("GC latency mode. SustainedLowLatency suppresses blocking gen2 collections (needs concurrent GC, Bonsai default).")]
        public GCLatencyMode GcLatencyMode { get; set; } = GCLatencyMode.SustainedLowLatency;

        [Description("Opt the process out of power throttling (EcoQoS / E-core steering when minimized) and, on Windows 11, " +
                     "out of background timer-resolution coalescing.")]
        public bool DisablePowerThrottling { get; set; } = true;

        [Description("Restrict the whole process to the performance (P) cores of a hybrid CPU. Process-wide mask, not a pin: " +
                     "threads may still migrate between P-cores. Off by default; measure before enabling.")]
        public bool RestrictToPerformanceCores { get; set; } = false;

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
                if (TimerResolution1ms)
                {
                    try
                    {
                        if (TimeBeginPeriod(1) == 0)
                        {
                            state.TimerResolution1ms = true;
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
                    if (RestrictToPerformanceCores)
                    {
                        long mask = FindPerformanceCoreMask();
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
