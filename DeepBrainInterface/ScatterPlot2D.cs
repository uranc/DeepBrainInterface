using System;
using System.ComponentModel;
using System.Drawing;
using System.Threading;
using System.Windows.Forms;
using Bonsai;
using OpenCV.Net;
using System.Reactive;
using System.Reactive.Linq;
using ZedGraph;

namespace DeepBrainInterface
{
    [Combinator]
    [Description("Real-time 2D scatter plot of 1×2 CV_32F Mats (X,Y) using ZedGraph in a dedicated UI thread.")]
    public class ScatterPlot2D : Combinator
    {
        [Description("Maximum number of points to retain in the plot (0 = no limit).")]
        public int Capacity { get; set; } = 0;

        [Description("Size of each scatter symbol.")]
        public int SymbolSize { get; set; } = 5;

        [Description("Automatically rescale axes to fit data.")]
        public bool AutoScale { get; set; } = true;

        // UI thread and synchronization
        Thread uiThread;
        ManualResetEvent handleCreated;

        // Form and ZedGraph elements
        Form form;
        ZedGraphControl zgc;
        GraphPane pane;
        PointPairList points;
        System.Windows.Forms.NumericUpDown capacityControl;

        public IObservable<Mat> Process(IObservable<Mat> source)
        {
            return Process(source, Observable.Never<Unit>());
        }

        public IObservable<Mat> Process(IObservable<Mat> source, IObservable<Unit> clear)
        {
            return Observable.Create<Mat>(observer =>
            {
                EnsurePlotWindow();

                var sourceSubscription = source.Do(mat =>
                {
                    if (mat.Rows != 1 || mat.Cols != 2)
                        throw new ArgumentException($"ScatterPlot2D expects a 1×2 Mat, got {mat.Rows}×{mat.Cols}.");

                    float x = (float)mat.GetReal(0, 0);
                    float y = (float)mat.GetReal(0, 1);

                    // Plot update on UI thread
                    form.BeginInvoke((Action)(() =>
                    {
                        // Add new point to series
                        points.Add(x, y);

                        // Keep only the specified number of points if capacity limit is set
                        if (Capacity > 0 && points.Count > Capacity)
                        {
                            // Remove oldest points to maintain capacity
                            while (points.Count > Capacity)
                            {
                                points.RemoveAt(0);
                            }
                        }

                        // Update auto-scale if needed
                        if (AutoScale) pane.AxisChange();

                        zgc.Invalidate();
                    }));
                }).Subscribe(observer);

                var clearSubscription = clear.Subscribe(_ =>
                {
                    form.BeginInvoke((Action)(() =>
                    {
                        points.Clear();
                        if (AutoScale) pane.AxisChange();
                        zgc.Invalidate();
                    }));
                });

                return new System.Reactive.Disposables.CompositeDisposable(sourceSubscription, clearSubscription);
            });
        }

        public override IObservable<TSource> Process<TSource>(IObservable<TSource> source)
        {
            var matSource = source as IObservable<Mat>;
            if (matSource == null)
            {
                throw new InvalidOperationException("Source must be an IObservable<Mat>.");
            }
            return (IObservable<TSource>)Process(matSource);
        }

        void EnsurePlotWindow()
        {
            if (handleCreated != null) return;
            handleCreated = new ManualResetEvent(false);

            uiThread = new Thread(() =>
            {
                Application.EnableVisualStyles();

                form = new Form { Text = "Scatter Plot 2D", Width = 600, Height = 450 };
                zgc = new ZedGraphControl { Dock = DockStyle.Fill };
                
                // Create panel for controls
                System.Windows.Forms.Panel controlPanel = new System.Windows.Forms.Panel
                {
                    Dock = DockStyle.Top,
                    Height = 40
                };

                // Capacity control
                System.Windows.Forms.Label capacityLabel = new System.Windows.Forms.Label
                {
                    Text = "Point Capacity:",
                    AutoSize = true,
                    Location = new System.Drawing.Point(10, 12)
                };
                
                capacityControl = new System.Windows.Forms.NumericUpDown
                {
                    Minimum = 0,
                    Maximum = 10000,
                    Value = Capacity,
                    Location = new System.Drawing.Point(105, 10),
                    Width = 80
                };
                capacityControl.ValueChanged += (s, e) =>
                {
                    Capacity = (int)capacityControl.Value;
                    
                    // If capacity decreased, trim points
                    if (Capacity > 0 && points != null && points.Count > Capacity)
                    {
                        while (points.Count > Capacity)
                        {
                            points.RemoveAt(0);
                        }
                        zgc.Invalidate();
                    }
                };
                
                controlPanel.Controls.Add(capacityLabel);
                controlPanel.Controls.Add(capacityControl);
                
                form.Controls.Add(controlPanel);
                form.Controls.Add(zgc);

                pane = zgc.GraphPane;
                pane.Title.Text = "2D Scatter";
                pane.XAxis.Title.Text = "X";
                pane.YAxis.Title.Text = "Y";
                pane.XAxis.Scale.MinAuto = AutoScale;
                pane.XAxis.Scale.MaxAuto = AutoScale;
                pane.YAxis.Scale.MinAuto = AutoScale;
                pane.YAxis.Scale.MaxAuto = AutoScale;

                // Using PointPairList instead of RollingPointPairList for simplicity
                points = new PointPairList();
                var curve = pane.AddCurve("Points", points, Color.Blue, SymbolType.Circle);
                curve.Line.IsVisible = true;
                curve.Symbol.Size = SymbolSize;
                curve.Symbol.Fill.Color = Color.Blue;

                // Signal when handle is created
                form.HandleCreated += (s, e) => handleCreated.Set();

                Application.Run(form);
            });
            uiThread.SetApartmentState(ApartmentState.STA);
            uiThread.IsBackground = true;
            uiThread.Start();

            // Wait for form handle before plotting
            handleCreated.WaitOne();
        }
    }
}
