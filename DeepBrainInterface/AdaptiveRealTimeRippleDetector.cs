using Bonsai;
using System;
using System.Collections.Generic;
using System.Reactive.Linq;
using NumSharp;
using System.Collections.Concurrent;
using System.Reactive.Concurrency;
using System.ComponentModel;

[Combinator(MethodName = "Process")]
public class AdaptiveRealTimeRippleDetector : Transform<NDArray, float[]>, IDisposable
{
    private readonly ConcurrentQueue<NDArray> rollingBuffer;
    private readonly object bufferLock = new object();
    private const int MaxQueueSize = 3;
    
    [Description("Buffer size for temporal window")]
    public int BufferSize { get; set; } = 50;

    [Description("Number of samples to skip between inferences")]
    public int SkipRate { get; set; } = 5;

    [Description("Maximum inference time in milliseconds before timeout")]
    public int TimeoutMs { get; set; } = 20;

    [Description("Enable state-based adaptation")]
    public bool UseStateAdaptation { get; set; } = false;

    private int samplingCounter = 0;

    public AdaptiveRealTimeRippleDetector()
    {
        rollingBuffer = new ConcurrentQueue<NDArray>();
        InitializeModel();
    }

    private bool ShouldSample()
    {
        samplingCounter++;
        if (samplingCounter >= SkipRate)
        {
            samplingCounter = 0;
            return true;
        }
        return false;
    }

    private bool ShouldSampleWithState(RippleStateInfo state)
    {
        if (state == null) return ShouldSample();
        
        SkipRate = DetermineSkipRate(state);
        return ShouldSample();
    }

    private int DetermineSkipRate(RippleStateInfo state)
    {
        // Dynamic skip rate based on state
        return state.State switch
        {
            RippleState.DefiniteRipple => 1,
            RippleState.PossibleRipple when state.Duration < 50 => 2,
            RippleState.PossibleRipple => 5,
            _ => SkipRate
        };
    }

    private void UpdateBuffer(NDArray input)
    {
        if (rollingBuffer.Count >= BufferSize) rollingBuffer.TryDequeue(out _);
        rollingBuffer.Enqueue(input);
    }

    private NDArray CreateInputBatch()
    {
        lock (bufferLock)
        {
            return np.stack(rollingBuffer.ToArray()).reshape((1, BufferSize, -1));
        }
    }

    public IObservable<float[]> Process(IObservable<NDArray> source) =>
        ProcessWithState(source.Select(x => (x, (RippleStateInfo)null)));

    public IObservable<float[]> ProcessWithState(IObservable<(NDArray, RippleStateInfo)> source)
    {
        return source
            .ObserveOn(ThreadPoolScheduler.Instance)
            .Where(pair => UseStateAdaptation ? 
                ShouldSampleWithState(pair.Item2) : 
                ShouldSample())
            .Select(input =>
            {
                lock (bufferLock)
                {
                    UpdateBuffer(input.Item1);
                    return CreateInputBatch();
                }
            })
            .Where(batch => rollingBuffer.Count < MaxQueueSize)
            .SelectMany(batch => 
                Observable.Start(() => RunModelInference(batch))
                    .SubscribeOn(ThreadPoolScheduler.Instance)
                    .Timeout(TimeSpan.FromMilliseconds(TimeoutMs)))
            .Catch<float[], TimeoutException>(ex => 
                Observable.Return(new float[8]))
            .ObserveOn(ThreadPoolScheduler.Instance);
    }

    private float[] RunModelInference(NDArray inputBatch)
    {
        var results = session.run(outputOperation.outputs[0], new FeedItem(inputOperation.outputs[0], inputBatch));
        return np.squeeze(results).ToArray<float>();
    }

    private void InitializeModel()
    {
        // Model setup logic here
    }

    public void Dispose()
    {
        // Dispose logic here
    }
}
