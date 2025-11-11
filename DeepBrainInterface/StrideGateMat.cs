using Bonsai;
using OpenCV.Net;
using System;
using System.ComponentModel;
using System.Reactive.Linq;

namespace DeepBrainInterface
{
    [Description("Passes every K-th Mat. Keep Buffer Skip=1; adjust K live.")]
    [WorkflowElementCategory(ElementCategory.Transform)]
    public sealed class StrideGateMat : Transform<Mat, Mat>
    {
        [Description("Emit one element every K inputs. 0 = pause, 1 = pass all.")]
        public int K { get; set; } = 1;

        int i;
        public override IObservable<Mat> Process(IObservable<Mat> source)
        {
            i = 0;
            return source.Where(_ =>
            {
                int k = K;
                if (k <= 0) return false; // paused
                if (k == 1) return true;  // fast path
                i++;
                if (i >= k) { i = 0; return true; }
                return false;
            });
        }
    }
}
