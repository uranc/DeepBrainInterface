using Bonsai;
using OpenCV.Net;
using Python.Runtime;
using System;
using System.ComponentModel;
using System.IO;
using System.Runtime.InteropServices;
using System.Reactive.Linq;
using System.Windows.Forms;
using System.Drawing.Design;

namespace DeepBrainInterface
{
    [Combinator]
    [Description("Projects high-dimensional input data into a low-dimensional embedding using ApproxAlignedUMAP in transform mode. " +
                 "A pretrained model is loaded once and used to project new data. Any NaN values in the input are replaced with 0.")]
    [WorkflowElementCategory(ElementCategory.Transform)]
    public class ProjectApproxUMAP
    {
        private string defaultHomeFolder;

        public ProjectApproxUMAP()
        {
            // Initialize with user's home directory
            defaultHomeFolder = Environment.GetFolderPath(Environment.SpecialFolder.UserProfile);
            pretrainedModelPath = Path.Combine(defaultHomeFolder, "Documents", "frozen_models", "umap_approx_model.pkl");
            pythonEnvPath = Path.Combine(defaultHomeFolder, "anaconda3");
        }

        [Description("If true, loads a pretrained model from PretrainedModelPath and uses its transform() method.")]
        public bool UsePretrained { get; set; } = true;

        private string pretrainedModelPath;
        
        [FileNameFilter("Pickle Files|*.pkl|All Files|*.*")]
        [Editor(typeof(FileNameEditor), typeof(UITypeEditor))]
        [Description("Path to the pretrained ApproxAlignedUMAP model (pickle file).")]
        public string PretrainedModelPath 
        { 
            get { return pretrainedModelPath; } 
            set 
            { 
                pretrainedModelPath = value;
                // Reset initialized state to force reloading the model
                initialized = false;
            }
        }

        private string pythonEnvPath;

        [Editor(typeof(FolderNameEditor), typeof(UITypeEditor))]
        [Description("Path to the Python/Anaconda environment folder containing the required packages.")]
        public string PythonEnvPath
        {
            get { return pythonEnvPath; }
            set
            {
                pythonEnvPath = value;
                // Reset initialized state to force reloading the environment
                initialized = false;
            }
        }

        [Description("Number of neighbors used (should match training if using pretrained mode).")]
        public int Neighbors { get; set; } = 15;

        [Description("Number of output embedding dimensions.")]
        public int OutputDimensions { get; set; } = 2;

        private bool initialized = false;
        private dynamic pretrainedModel = null;
        private dynamic npModule = null;
        private dynamic approxUmapModule = null;

        private void EnsureInitialized()
        {
            if (initialized) return;

            string envPath = PythonEnvPath;
            string pythonDll = Path.Combine(envPath, "python310.dll");

            if (!File.Exists(pythonDll))
                throw new InvalidOperationException("python310.dll not found in the specified Anaconda environment.");

            Environment.SetEnvironmentVariable("PYTHONHOME", envPath);
            Environment.SetEnvironmentVariable("PYTHONPATH",
                $"{envPath};{envPath}\\Lib;{envPath}\\Lib\\site-packages;{envPath}\\DLLs");
            string currentPath = Environment.GetEnvironmentVariable("PATH");
            Environment.SetEnvironmentVariable("PATH", $"{envPath};{envPath}\\Library\\bin;{currentPath}");

            Runtime.PythonDLL = pythonDll;
            PythonEngine.PythonHome = envPath;
            PythonEngine.PythonPath = Environment.GetEnvironmentVariable("PYTHONPATH");

            PythonEngine.Initialize();
            using (Py.GIL())
            {
                npModule = Py.Import("numpy");
                approxUmapModule = Py.Import("approx_umap");
            }

            if (UsePretrained)
            {
                if (!File.Exists(PretrainedModelPath))
                    throw new FileNotFoundException($"Pretrained model file not found: {PretrainedModelPath}");

                using (Py.GIL())
                {
                    dynamic pickle = Py.Import("pickle");
                    dynamic builtins = Py.Import("builtins");
                    dynamic file = builtins.open(PretrainedModelPath, "rb");
                    pretrainedModel = pickle.load(file);
                    file.close();
                }
            }

            initialized = true;
        }

        public IObservable<Mat> Process(IObservable<Mat> source)
        {
            return source.Select(input =>
            {
                EnsureInitialized();
                float[,] data = ConvertMatToArray(input);
                float[,] embedding = ApplyUMAP(data);
                Mat output = ConvertArrayToMat(embedding);
                return output;
            });
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

        private float[,] ApplyUMAP(float[,] data)
        {
            using (Py.GIL())
            {
                // Convert C# array to a numpy array (dtype float32)
                dynamic npArray = npModule.array(data, dtype: npModule.float32);

                // Replace any NaN values with 0 (minimal imputation)
                if (npModule.any(npModule.isnan(npArray)).ToString().ToLower().Contains("true"))
                {
                    // np.nan_to_num replaces nan with 0 by default
                    npArray = npModule.nan_to_num(npArray);
                }
                float[,] result;
                // Use the pretrained model's transform method.
                dynamic embedding = pretrainedModel.transform(npArray);
                result = ConvertEmbeddingToArray(embedding);
                return result;
            }
        }

        private float[,] ConvertEmbeddingToArray(dynamic embedding)
        {
            int rows = (int)embedding.shape[0];
            int cols = (int)embedding.shape[1];
            float[,] result = new float[rows, cols];

            dynamic npFlat = embedding.flatten().tolist();
            float[] flatArray = ((float[])npFlat.AsManagedObject(typeof(float[])));

            for (int i = 0; i < rows; i++)
                for (int j = 0; j < cols; j++)
                    result[i, j] = flatArray[i * cols + j];

            return result;
        }
    }

    // Custom attribute to specify file filter for FileNameEditor
    [AttributeUsage(AttributeTargets.Property)]
    public class FileNameFilterAttribute : Attribute
    {
        public FileNameFilterAttribute(string filter)
        {
            Filter = filter;
        }

        public string Filter { get; }
    }

    // Custom FileNameEditor to use our filter attribute
    public class FileNameEditor : UITypeEditor
    {
        public override UITypeEditorEditStyle GetEditStyle(ITypeDescriptorContext context)
        {
            return UITypeEditorEditStyle.Modal;
        }

        public override object EditValue(ITypeDescriptorContext context, IServiceProvider provider, object value)
        {
            if (context == null || provider == null)
                return value;

            using (OpenFileDialog dialog = new OpenFileDialog())
            {
                // Set default filter
                dialog.Filter = "All Files (*.*)|*.*";
                
                // Check for custom filter attribute
                FileNameFilterAttribute filterAttr = (FileNameFilterAttribute)context.PropertyDescriptor.Attributes[typeof(FileNameFilterAttribute)];
                if (filterAttr != null)
                {
                    dialog.Filter = filterAttr.Filter;
                }

                // Set initial directory if value is not empty
                if (value != null && !string.IsNullOrEmpty(value.ToString()))
                {
                    try
                    {
                        string initialDir = Path.GetDirectoryName(value.ToString());
                        if (Directory.Exists(initialDir))
                        {
                            dialog.InitialDirectory = initialDir;
                            dialog.FileName = Path.GetFileName(value.ToString());
                        }
                    }
                    catch { /* Ignore directory resolution errors */ }
                }

                // Show dialog and return selected file
                if (dialog.ShowDialog() == DialogResult.OK)
                {
                    return dialog.FileName;
                }
            }

            return value;
        }
    }

    // Folder name editor for selecting directories
    public class FolderNameEditor : UITypeEditor
    {
        public override UITypeEditorEditStyle GetEditStyle(ITypeDescriptorContext context)
        {
            return UITypeEditorEditStyle.Modal;
        }

        public override object EditValue(ITypeDescriptorContext context, IServiceProvider provider, object value)
        {
            if (context == null || provider == null)
                return value;

            using (FolderBrowserDialog dialog = new FolderBrowserDialog())
            {
                // Set description
                dialog.Description = "Select Python/Anaconda environment folder";
                
                // Set initial directory if value is not empty
                if (value != null && !string.IsNullOrEmpty(value.ToString()))
                {
                    try
                    {
                        if (Directory.Exists(value.ToString()))
                        {
                            dialog.SelectedPath = value.ToString();
                        }
                    }
                    catch { /* Ignore directory resolution errors */ }
                }

                // Show dialog and return selected folder
                if (dialog.ShowDialog() == DialogResult.OK)
                {
                    return dialog.SelectedPath;
                }
            }

            return value;
        }
    }
}
