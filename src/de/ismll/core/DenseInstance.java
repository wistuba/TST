package de.ismll.core;

import java.util.HashMap;

public class DenseInstance extends Instance
{
	private static HashMap<Integer, int[]> cachedIndices = new HashMap<Integer, int[]>();

	public DenseInstance(double target, double[] values)
	{
		super(target, values);
		if(!cachedIndices.containsKey(values.length))
		{
			int[] indices = new int[values.length];
			for(int i = 0; i < indices.length; i++)
				indices[i] = i;
			cachedIndices.put(indices.length, indices);
		}
	}

	@Override
	public double getValue(int index)
	{
		return this.values[index];
	}

	@Override
	public double[] getValues()
	{
		return this.values;
	}

	@Override
	public int[] getKeys()
	{
		return cachedIndices.get(this.values.length);
	}

	@Override
	public void setValue(double value, int index)
	{
		this.values[index] = value;
	}
}
