using Bonsai;
using OpenCV.Net;
using Python.Runtime;
using System;
using System.ComponentModel;
using System.Drawing.Design;
using System.IO;
using System.Reactive.Linq;
using System.Runtime.InteropServices;
using System.Windows.Forms.Design; // Ensure System.Design is referenced

namespace DeepBrainInterface
{
    [Combinator]
    [Description("Projects high-dimensional input data into a low-dimensional embedding using ApproxAlignedUMAP in transform mode.")]
    [WorkflowElementCategory(ElementCategory.Transform)]
    public class ProjectApproxUMAP
    {
        private string defaultHomeFolder;
        private string pretrainedModelPath;
        private string pythonEnvPath;

        // Python objects held globally for the session
        private bool initialized = false;
        private dynamic pretrainedModel = null;
        private dynamic npModule = null;

        public ProjectApproxUMAP()
        {
            // Initialize with user's home directory defaults
            defaultHomeFolder = Environment.GetFolderPath(Environment.SpecialFolder.UserProfile);
            pretrainedModelPath = Path.Combine(defaultHomeFolder, "Documents", "frozen_models", "umap_approx_model.pkl");
            pythonEnvPath = Path.Combine(defaultHomeFolder, "anaconda3");
        }

        [Description("If true, loads a pretrained model from PretrainedModelPath.")]
        public bool UsePretrained { get; set; } = true;

        [FileNameFilter("Pickle Files|*.pkl|All Files|*.*")]
        [Editor(typeof(FileNameEditor), typeof(UITypeEditor))]
        [Description("Path to the pretrained ApproxAlignedUMAP model (pickle file).")]
        public string PretrainedModelPath
        {
            get { return pretrainedModelPath; }
            set
            {
                pretrainedModelPath = value;
                initialized = false;
            }
        }

        [Editor(typeof(FolderNameEditor), typeof(UITypeEditor))]
        [Description("Path to the Python/Anaconda environment folder containing the required packages.")]
        public string PythonEnvPath
        {
            get { return pythonEnvPath; }
            set
            {
                pythonEnvPath = value;
                initialized = false;
            }
        }

        [Description("Number of neighbors used (should match training if using pretrained mode).")]
        public int Neighbors { get; set; } = 15;

        [Description("Number of output embedding dimensions.")]
        public int OutputDimensions { get; set; } = 2;

        public IObservable<Mat> Process(IObservable<Mat> source)
        {
            // Defer allows us to run initialization logic once per workflow start
            return Observable.Defer(() =>
            {
                EnsureInitialized();

                return source.Select(input =>
                {
                    float[,] data = ConvertMatToArray(input);
                    float[,] embedding = ApplyUMAP(data);
                    return ConvertArrayToMat(embedding);
                });
            });
        }

        private void EnsureInitialized()
        {
            if (initialized) return;

            string envPath = PythonEnvPath;
            // Note: If you upgrade conda, this might change to python311.dll
            string pythonDll = Path.Combine(envPath, "python310.dll");

            if (!File.Exists(pythonDll))
                throw new InvalidOperationException($"python310.dll not found in {envPath}");

            // Set Environment Variables required for Python.Runtime to find the environment
            Environment.SetEnvironmentVariable("PYTHONHOME", envPath);
            Environment.SetEnvironmentVariable("PYTHONPATH", $"{envPath};{envPath}\\Lib;{envPath}\\Lib\\site-packages;{envPath}\\DLLs");
            string currentPath = Environment.GetEnvironmentVariable("PATH");
            Environment.SetEnvironmentVariable("PATH", $"{envPath};{envPath}\\Library\\bin;{currentPath}");

            Runtime.PythonDLL = pythonDll;
            if (!PythonEngine.IsInitialized)
            {
                PythonEngine.Initialize();
                PythonEngine.BeginAllowThreads(); // Allow other threads to use Python
            }

            using (Py.GIL())
            {
                npModule = Py.Import("numpy");
                // approx_umap must be installed in the conda env
                Py.Import("approx_umap");

                if (UsePretrained)
                {
                    if (!File.Exists(PretrainedModelPath))
                        throw new FileNotFoundException($"Model file not found: {PretrainedModelPath}");

                    dynamic pickle = Py.Import("pickle");
                    dynamic builtins = Py.Import("builtins");

                    // Safe file opening in Python
                    using (dynamic file = builtins.open(PretrainedModelPath, "rb"))
                    {
                        pretrainedModel = pickle.load(file);
                    }
                }
            }
            initialized = true;
        }

        private float[,] ApplyUMAP(float[,] data)
        {
            using (Py.GIL())
            {
                // CRITICAL: Wrap PyObjects in 'using' to decrement ref counts and avoid memory leaks
                using (dynamic npArray = npModule.array(data, dtype: npModule.float32))
                {
                    // Check for NaNs safely
                    dynamic hasNanObj = npModule.any(npModule.isnan(npArray));
                    bool hasNan = (bool)hasNanObj.AsManagedObject(typeof(bool));
                    hasNanObj.Dispose(); // Clean up the intermediate bool object

                    if (hasNan)
                    {
                        using (dynamic cleanedArray = npModule.nan_to_num(npArray))
                        using (dynamic embedding = pretrainedModel.transform(cleanedArray))
                        {
                            return ConvertEmbeddingToArray(embedding);
                        }
                    }
                    else
                    {
                        using (dynamic embedding = pretrainedModel.transform(npArray))
                        {
                            return ConvertEmbeddingToArray(embedding);
                        }
                    }
                }
            }
        }

        private float[,] ConvertEmbeddingToArray(dynamic embedding)
        {
            int rows = (int)embedding.shape[0];
            int cols = (int)embedding.shape[1];
            float[,] result = new float[rows, cols];

            using (dynamic npFlat = embedding.flatten())
            using (dynamic npList = npFlat.tolist())
            {
                float[] flatArray = ((float[])npList.AsManagedObject(typeof(float[])));
                for (int i = 0; i < rows; i++)
                    for (int j = 0; j < cols; j++)
                        result[i, j] = flatArray[i * cols + j];
            }
            return result;
        }

        private float[,] ConvertMatToArray(Mat input)
        {
            int rows = input.Rows, cols = input.Cols;
            float[] flatData = new float[rows * cols];
            Marshal.Copy(input.Data, flatData, 0, flatData.Length);
            float[,] data = new float[rows, cols];
            for (int i = 0; i < rows; i++)
                for (int j = 0; j < cols; j++)
                    data[i, j] = flatData[i * cols + j];
            return data;
        }

        private Mat ConvertArrayToMat(float[,] data)
        {
            int rows = data.GetLength(0), cols = data.GetLength(1);
            Mat mat = new Mat(rows, cols, Depth.F32, 1);
            float[] flat = new float[rows * cols];
            for (int i = 0; i < rows; i++)
                for (int j = 0; j < cols; j++)
                    flat[i * cols + j] = data[i, j];
            Marshal.Copy(flat, 0, mat.Data, flat.Length);
            return mat;
        }
    }

    // --- Editor Helper Classes ---

    [AttributeUsage(AttributeTargets.Property)]
    public class FileNameFilterAttribute : Attribute
    {
        public FileNameFilterAttribute(string filter) { Filter = filter; }
        public string Filter { get; }
    }

    public class FileNameEditor : UITypeEditor
    {
        public override UITypeEditorEditStyle GetEditStyle(ITypeDescriptorContext context) => UITypeEditorEditStyle.Modal;

        public override object EditValue(ITypeDescriptorContext context, IServiceProvider provider, object value)
        {
            using (var dialog = new System.Windows.Forms.OpenFileDialog())
            {
                var attr = context?.PropertyDescriptor?.Attributes[typeof(FileNameFilterAttribute)] as FileNameFilterAttribute;
                dialog.Filter = attr?.Filter ?? "All Files|*.*";

                if (value != null && !string.IsNullOrEmpty(value.ToString()))
                {
                    try { dialog.InitialDirectory = Path.GetDirectoryName(value.ToString()); } catch { }
                }

                if (dialog.ShowDialog() == System.Windows.Forms.DialogResult.OK) return dialog.FileName;
            }
            return value;
        }
    }

    public class FolderNameEditor : UITypeEditor
    {
        public override UITypeEditorEditStyle GetEditStyle(ITypeDescriptorContext context) => UITypeEditorEditStyle.Modal;

        public override object EditValue(ITypeDescriptorContext context, IServiceProvider provider, object value)
        {
            using (var dialog = new System.Windows.Forms.FolderBrowserDialog())
            {
                if (value != null && Directory.Exists(value.ToString())) dialog.SelectedPath = value.ToString();
                if (dialog.ShowDialog() == System.Windows.Forms.DialogResult.OK) return dialog.SelectedPath;
            }
            return value;
        }
    }
}