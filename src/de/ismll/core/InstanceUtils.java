package de.ismll.core;


public abstract class InstanceUtils
{
	public static DenseInstance createDenseInstance(double target, double... values)
	{
		return new DenseInstance(target, values);
	}

	public static SparseInstance createSparseInstance(double target, double[] values, int[] keys)
	{
		return new SparseInstance(target, keys, values);
	}

	public static double dotProduct(Instance i1, Instance i2)
	{
		double result = 0;
		int index1 = 0, index2 = 0;
		int[] keys1 = i1.getKeys(), keys2 = i2.getKeys();
		double[] values1 = i1.getValues(), values2 = i2.getValues();
		while(index1 < keys1.length && index2 < keys2.length)
		{
			if(keys1[index1] < keys2[index2])
				index1++;
			else if(keys1[index1] > keys2[index2])
				index2++;
			else
				result += values1[index1++] * values2[index2++];
		}
		return result;
	}

	public static double euclideanDistance(Instance i1, Instance i2)
	{
		return euclideanDistance(i1, i2, Integer.MAX_VALUE);
	}

	/**
	 * Euclidean distance between two instances using only the first m attributes.
	 */
	public static double euclideanDistance(Instance i1, Instance i2, int m)
	{
		double result = 0;
		int index1 = 0, index2 = 0;
		int[] keys1 = i1.getKeys(), keys2 = i2.getKeys();
		double[] values1 = i1.getValues(), values2 = i2.getValues();
		while((index1 < keys1.length || index2 < keys2.length) && (index1 < keys1.length && keys1[index1] < m || index2 < keys2.length && keys2[index2] < m))
		{
			if(index1 < keys1.length && keys1[index1] < m && (index2 >= keys2.length || keys2[index2] > keys1[index1]))
				result += Math.pow(values1[index1++], 2);
			else if(index2 < keys2.length && keys2[index2] < m && (index1 >= keys1.length || keys1[index1] > keys2[index2]))
				result += Math.pow(values2[index2++], 2);
			else if(index1 < keys1.length && index2 < keys2.length && keys1[index1] < m && keys2[index2] < m )
				result += Math.pow(values1[index1++] - values2[index2++], 2);
		}
		return Math.sqrt(result);
	}

	public static int getRank(Instances instances, Instance instance)
	{
		double target = instance.target();
		int rank = 1;
		for(int j = 0; j < instances.numInstances(); j++)
		{
			if(instances.instance(j).target() > target)
				rank++;
		}
		return rank;
	}
}
