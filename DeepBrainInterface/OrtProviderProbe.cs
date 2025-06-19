using Bonsai;
using Microsoft.ML.OnnxRuntime;
using OpenCV.Net;
using System;
using System.ComponentModel;
using System.Drawing.Design;
using System.Reactive.Linq;
using System.Windows.Forms.Design;

namespace DeepBrainInterface
{
    /// <summary>
    /// One‐shot probe that attempts to append the CUDA and TensorRT
    /// execution providers and logs which ones succeeded.
    /// </summary>
    [Combinator]
    [Description("Probes GPU + TensorRT availability and logs which providers are available.")]
    [WorkflowElementCategory(ElementCategory.Transform)]
    public class OrtProviderProbe : Transform<Mat, Mat>
    {
        [Editor(typeof(FileNameEditor), typeof(UITypeEditor))]
        [Description("Optional: path to any ONNX model file (not actually loaded by this node).")]
        public string ModelPath { get; set; } = "";

        bool _logged;

        public override IObservable<Mat> Process(IObservable<Mat> source)
        {
            return source.Select(mat =>
            {
                if (!_logged)
                {
                    // Prepare session options
                    var options = new SessionOptions();

                    // Try CUDA
                    bool cudaAvailable;
                    try
                    {
                        options.AppendExecutionProvider_CUDA(0);
                        cudaAvailable = true;
                    }
                    catch (Exception)
                    {
                        cudaAvailable = false;
                    }

                    // Try TensorRT
                    bool trtAvailable;
                    try
                    {
                        options.AppendExecutionProvider_Tensorrt(0);
                        trtAvailable = true;
                    }
                    catch (Exception)
                    {
                        trtAvailable = false;
                    }

                    Console.WriteLine($"CUDA available: {cudaAvailable}, TensorRT available: {trtAvailable}");
                    _logged = true;
                }

                // Pass the data through unmodified
                return mat;
            });
        }
    }
}
