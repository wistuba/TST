package de.ismll.core;

import java.util.Arrays;

public class SparseInstance extends Instance
{
	protected final int[] keys;

	SparseInstance(double target, int[] keys, double[] values)
	{
		super(target, values);
		this.keys = keys;
	}

	@Override
	public double getValue(int index)
	{
		int i = Arrays.binarySearch(this.keys, index);
		if(i >= 0)
		{
			return this.values[i];
		}
		else
		{
			return 0;
		}
	}

	@Override
	public double[] getValues()
	{
		return this.values;
	}

	@Override
	public int[] getKeys()
	{
		return this.keys;
	}

	@Override
	public void setValue(double value, int index)
	{
		throw new IllegalArgumentException("Operation not supported.");
	}
	
	@Override
	public String toString()
	{
		StringBuilder sb = new StringBuilder();
		sb.append(this.target());
		for(int i = 0; i < this.keys.length; i++)
			sb.append(" ").append(this.keys[i]).append(":").append(this.values[i]);
		return sb.toString();
	}
}
