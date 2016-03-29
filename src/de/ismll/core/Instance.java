package de.ismll.core;

import java.util.Arrays;

public abstract class Instance
{
	private double target;

	protected final double[] values;

	public Instance(double target, double[] values)
	{
		this.target = target;
		this.values = Arrays.copyOf(values, values.length);
	}

	public abstract double getValue(int index);

	public abstract double[] getValues();
	
	public abstract void setValue(double value, int index);

	public abstract int[] getKeys();

	public void setTarget(double target)
	{
		this.target = target;
	}
	
	

	public double target()
	{
		return this.target;
	}
}
