using DeepBrainInterface;
using System;
using Xunit;

public class RippleDisplayBlockTests
{
    [Fact]
    public void EmptyMapAndChannelsIsIdentity()
    {
        Assert.Equal(new[] { 0, 1, 2, 3 }, RippleDisplayBlock.ResolveRows(new int[0], new int[0], 4));
        Assert.Equal(new[] { 0, 1, 2, 3 }, RippleDisplayBlock.ResolveRows(null, null, 4));
    }

    [Fact]
    public void ChannelsIndexIntoTheMappedOrder()
    {
        var map = new[] { 3, 1, 0, 2 };                                              // display row k shows input row map[k]
        Assert.Equal(map, RippleDisplayBlock.ResolveRows(map, new int[0], 4));
        Assert.Equal(new[] { 0, 2 }, RippleDisplayBlock.ResolveRows(map, new[] { 2, 3 }, 4));   // mapped rows 2,3 = input rows 0,2
        Assert.Equal(new[] { 1, 3 }, RippleDisplayBlock.ResolveRows(new int[0], new[] { 1, 3 }, 4));
    }

    [Fact]
    public void RejectsBadMapsAndIndices()
    {
        Assert.Throws<InvalidOperationException>(() => RippleDisplayBlock.ResolveRows(new[] { 0, 1 }, null, 4));        // wrong length
        Assert.Throws<InvalidOperationException>(() => RippleDisplayBlock.ResolveRows(new[] { 0, 0, 1, 2 }, null, 4));  // duplicate
        Assert.Throws<InvalidOperationException>(() => RippleDisplayBlock.ResolveRows(new[] { 0, 1, 2, 9 }, null, 4));  // out of range
        Assert.Throws<InvalidOperationException>(() => RippleDisplayBlock.ResolveRows(null, new[] { 4 }, 4));          // channel out of range
    }
}
