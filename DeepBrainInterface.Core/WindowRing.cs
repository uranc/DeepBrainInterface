using System;

namespace DeepBrainInterface.Core
{
    /// <summary>Ring of the last TimePoints channel vectors, sample-major. Each push is written twice (at i and i + TimePoints)
    /// into a 2*TimePoints backing store, so the current window is always one contiguous block and CopyWindow is a single copy.</summary>
    public sealed class WindowRing
    {
        readonly int timePoints, channels;
        readonly float[] buf;   // 2 * timePoints * channels
        int head;               // next write slot in [0, timePoints)
        int count;

        public WindowRing(int timePoints, int channels)
        {
            if (timePoints < 1) throw new ArgumentOutOfRangeException(nameof(timePoints));
            if (channels < 1) throw new ArgumentOutOfRangeException(nameof(channels));
            this.timePoints = timePoints; this.channels = channels;
            buf = new float[2 * timePoints * channels];
        }

        public int TimePoints { get { return timePoints; } }
        public int Channels { get { return channels; } }
        public bool IsFull { get { return count >= timePoints; } }

        public void Push(float[] sample)
        {
            int o = head * channels;
            Array.Copy(sample, 0, buf, o, channels);
            Array.Copy(sample, 0, buf, o + timePoints * channels, channels);
            head = head + 1 == timePoints ? 0 : head + 1;
            if (count < timePoints) count++;
        }

        /// <summary>Copies the last TimePoints samples, oldest first, to dst[dstOffset..]. The oldest sample sits at slot `head`
        /// (the next write position), and slots head..head+timePoints-1 are contiguous in the doubled store.</summary>
        public void CopyWindow(float[] dst, int dstOffset)
        {
            Array.Copy(buf, head * channels, dst, dstOffset, timePoints * channels);
        }

        public void Reset() { head = 0; count = 0; Array.Clear(buf, 0, buf.Length); }
    }
}
