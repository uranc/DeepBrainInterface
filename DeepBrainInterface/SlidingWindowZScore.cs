using Bonsai;
using DeepBrainInterface.Core;
using OpenCV.Net;
using System;
using System.ComponentModel;
using System.Reactive.Linq;

namespace DeepBrainInterface
{
    [Combinator]
    [Description("Exact Rectangular Sliding Window Z-score. Matches Python training environment. Shell over DeepBrainInterface.Core.ZScoreStage (1.1.78).")]
    [WorkflowElementCategory(ElementCategory.Transform)]
    public class SlidingWindowZScore
    {
        [Description("Exact window size in samples.")]
        public int WindowSize { get; set; } = 1250;

        [Description("Epsilon to prevent division by zero.")]
        public float Eps { get; set; } = 1e-8f;

        public IObservable<Mat> Process(IObservable<Mat> source)
        {
            return Observable.Defer(() =>
            {
                ZScoreStage stage = null; float[] vec = null; Mat outputMat = null;
                return source.Select(input =>
                {
                    int ch = input.Rows, nIn = input.Cols;
                    if (stage == null || stage.Channels != ch) { stage = new ZScoreStage(ch, WindowSize, Eps); vec = new float[ch]; }
                    if (outputMat == null || outputMat.Rows != ch || outputMat.Cols != nIn) outputMat = new Mat(ch, nIn, Depth.F32, 1);
                    Depth inDepth = input.Depth;
                    if (inDepth != Depth.F32 && inDepth != Depth.U16 && inDepth != Depth.S16)
                        throw new InvalidOperationException("SlidingWindowZScore: input depth must be F32, U16 or S16, got " + inDepth);
                    unsafe
                    {
                        byte* inBytes = (byte*)input.Data.ToPointer(); byte* outBytes = (byte*)outputMat.Data.ToPointer();
                        int inStep = input.Step, outStep = outputMat.Step;
                        for (int t = 0; t < nIn; t++)
                        {
                            for (int c = 0; c < ch; c++)
                            {
                                byte* src = inBytes + c * inStep;
                                vec[c] = inDepth == Depth.F32 ? *(float*)(src + t * 4) : inDepth == Depth.U16 ? (float)*(ushort*)(src + t * 2) : (float)*(short*)(src + t * 2);
                            }
                            stage.Update(vec);
                            for (int c = 0; c < ch; c++) *(float*)(outBytes + c * outStep + t * sizeof(float)) = vec[c];
                        }
                    }
                    return outputMat;
                });
            });
        }
    }
}
