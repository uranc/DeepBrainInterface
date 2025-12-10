using Bonsai;
using OpenCV.Net;
using System;
using System.ComponentModel;
using System.Numerics;
using System.Reactive.Linq;

namespace DeepBrainInterface
{
    [Description("Converts Vector3 or Quaternion data to OpenCV Mat format")]
    public class BNOToMatVector3 : Transform<Vector3, Mat>
    {
        public override IObservable<Mat> Process(IObservable<Vector3> source)
        {
            return source.Select(v =>
            {
                var m = new Mat(3, 1, Depth.F32, 1);
                m.SetReal(0, 0, v.X);
                m.SetReal(1, 0, v.Y);
                m.SetReal(2, 0, v.Z);
                return m;
            });
        }
    }

    [Description("Converts Quaternion data to OpenCV Mat format")]
    public class BNOToMatQuaternion : Transform<Quaternion, Mat>
    {
        public override IObservable<Mat> Process(IObservable<Quaternion> source)
        {
            return source.Select(q =>
            {
                var m = new Mat(4, 1, Depth.F32, 1);
                m.SetReal(0, 0, q.X);
                m.SetReal(1, 0, q.Y);
                m.SetReal(2, 0, q.Z);
                m.SetReal(3, 0, q.W);
                return m;
            });
        }
    }
}
