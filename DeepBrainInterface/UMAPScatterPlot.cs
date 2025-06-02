using Bonsai;
using OpenCV.Net;
using ScottPlot;
using ScottPlot.WinForms;
using System;
using System.ComponentModel;
using System.Reactive;
using System.Reactive.Linq;
using System.Runtime.InteropServices;
using System.Windows.Forms;

namespace DeepBrainInterface
{
    [Combinator]
    [Description("Plots 2D Mat embeddings (Nx2) in real-time as a scatter plot.")]
    [WorkflowElementCategory(ElementCategory.Sink)]
    public class UMAP2DScatterPlot
    {
        private readonly FormsPlot formsPlot;
        private readonly Form plotForm;

        [Description("Marker size of the plotted points.")]
        public float MarkerSize { get; set; } = 5f;

        public UMAP2DScatterPlot()
        {
            formsPlot = new FormsPlot { Dock = DockStyle.Fill };
            plotForm = new Form
            {
                Width = 600,
                Height = 600,
                Text = "UMAP 2D Scatter Plot"
            };
            plotForm.Controls.Add(formsPlot);
        }

        public IObservable<Unit> Process(IObservable<Mat> source)
        {
            return Observable.Create<Unit>(observer =>
            {
                if (!plotForm.IsHandleCreated || plotForm.IsDisposed)
                {
                    plotForm.Show();
                }

                return source.Subscribe(data =>
                {
                    if (data.Cols != 2)
                        throw new InvalidOperationException("Input Mat must have exactly 2 columns for 2D scatter plot.");

                    int rows = data.Rows;
                    float[] buffer = new float[rows * 2];
                    Marshal.Copy(data.Data, buffer, 0, buffer.Length);

                    double[] xs = new double[rows];
                    double[] ys = new double[rows];
                    for (int i = 0; i < rows; i++)
                    {
                        xs[i] = buffer[i * 2];
                        ys[i] = buffer[i * 2 + 1];
                    }

                    formsPlot.Plot.Clear();
                    formsPlot.Plot.Add.Scatter(xs, ys).MarkerSize = MarkerSize;
                    //formsPlot.Plot.AxisAuto();
                    formsPlot.Refresh();

                    observer.OnNext(Unit.Default);
                },
                ex => observer.OnError(ex),
                () => observer.OnCompleted());
            });
        }
    }
}
