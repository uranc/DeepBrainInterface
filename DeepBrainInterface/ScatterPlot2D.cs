using Bonsai;
using OpenCV.Net;
using System;
using System.ComponentModel;
using System.Drawing;
using System.Reactive.Disposables;
using System.Reactive.Linq;
using System.Threading;
using System.Windows.Forms;
using ZedGraph;

namespace DeepBrainInterface
{
    // CHANGE: Inherit from Sink<Mat> instead of Combinator.
    // This fixes the "does not implement inherited abstract member" error automatically.
    [Combinator]
    [Description("Real-time 2D scatter plot of 1x2 CV_32F Mats (X,Y). Uses a circular buffer for high performance.")]
    public class ScatterPlot2D : Sink<Mat>
    {
        [Description("Maximum number of points to retain in the plot.")]
        public int Capacity { get; set; } = 1000;

        [Description("Size of each scatter symbol.")]
        public int SymbolSize { get; set; } = 5;

        [Description("Automatically rescale axes to fit data.")]
        public bool AutoScale { get; set; } = true;

        [Description("X Axis Label")]
        public string XLabel { get; set; } = "UMAP 1";

        [Description("Y Axis Label")]
        public string YLabel { get; set; } = "UMAP 2";

        // Internal UI references
        private Form form;
        private ZedGraphControl zgc;
        private RollingPointPairList points;
        private System.Windows.Forms.Timer renderTimer;

        // CHANGE: Added 'override' keyword because we are now satisfying Sink<Mat>'s contract
        public override IObservable<Mat> Process(IObservable<Mat> source)
        {
            return Observable.Create<Mat>(observer =>
            {
                var handleCreated = new ManualResetEvent(false);

                // 1. Create UI Thread
                var uiThread = new Thread(() =>
                {
                    form = new Form
                    {
                        Text = "Real-time Scatter Plot",
                        Width = 600,
                        Height = 600
                    };

                    zgc = new ZedGraphControl { Dock = DockStyle.Fill };
                    form.Controls.Add(zgc);

                    var pane = zgc.GraphPane;
                    pane.Title.Text = "Embedding Space";
                    pane.XAxis.Title.Text = XLabel;
                    pane.YAxis.Title.Text = YLabel;

                    points = new RollingPointPairList(Capacity);

                    LineItem curve = pane.AddCurve("Live Data", points, Color.Blue, SymbolType.Circle);
                    curve.Line.IsVisible = false;
                    curve.Symbol.Size = SymbolSize;
                    curve.Symbol.Fill = new Fill(Color.FromArgb(150, Color.Blue));
                    curve.Symbol.Border.IsVisible = false;

                    // Render Timer (30Hz)
                    renderTimer = new System.Windows.Forms.Timer();
                    renderTimer.Interval = 33;
                    renderTimer.Tick += (s, e) =>
                    {
                        if (form.IsDisposed || zgc.IsDisposed) return;
                        if (AutoScale) pane.AxisChange();
                        zgc.Invalidate();
                    };
                    renderTimer.Start();

                    form.FormClosing += (s, e) =>
                    {
                        if (e.CloseReason == CloseReason.UserClosing)
                        {
                            e.Cancel = true;
                            form.Hide();
                        }
                    };

                    form.HandleCreated += (s, e) => handleCreated.Set();

                    Application.Run(form);
                });

                uiThread.SetApartmentState(ApartmentState.STA);
                uiThread.IsBackground = true;
                uiThread.Start();

                handleCreated.WaitOne();

                // 2. Data Subscription
                var sourceSub = source.Do(mat =>
                {
                    if (mat.Rows != 1 || mat.Cols != 2) return;

                    float x = (float)mat.GetReal(0, 0);
                    float y = (float)mat.GetReal(0, 1);

                    if (!form.IsDisposed)
                    {
                        form.BeginInvoke((Action)(() =>
                        {
                            if (points == null) return;

                            // Handle dynamic capacity change
                            if (points.Capacity != Capacity)
                            {
                                var newPoints = new RollingPointPairList(Capacity);
                                newPoints.Add(points);
                                points = newPoints;
                                zgc.GraphPane.CurveList[0].Points = points;
                            }

                            points.Add(x, y);
                        }));
                    }
                }).Subscribe(observer);

                // 3. Cleanup Logic
                return Disposable.Create(() =>
                {
                    sourceSub.Dispose();
                    if (form != null && !form.IsDisposed)
                    {
                        form.Invoke((Action)(() =>
                        {
                            renderTimer?.Stop();
                            renderTimer?.Dispose();
                            form.Dispose();
                        }));
                    }
                });
            });
        }
    }
}