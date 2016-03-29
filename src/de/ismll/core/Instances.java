package de.ismll.core;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileOutputStream;
import java.io.FileReader;
import java.io.IOException;
import java.io.OutputStreamWriter;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Iterator;

public class Instances implements Iterable<Instance>
{
	protected ArrayList<Instance> instances = new ArrayList<Instance>();

	private int numValues;

	public Instances(int numValues)
	{
		this.numValues = numValues;
	}
	
	public Instances(String filename) throws IOException
	{
		this(new File(filename), " ");
	}
	
	public Instances(String filename, String delimiter) throws IOException
	{
		this(new File(filename), delimiter);
	}
	
	public Instances(File file) throws IOException {
		this(file," ");
	}

	public Instances(File file, String delimiter) throws IOException
	{
		BufferedReader br = new BufferedReader(new FileReader(file));
		String line;
		boolean firstLine = true, sparse = false;
		int maxKey = 0;
		while((line = br.readLine()) != null)
		{
			if(firstLine)
			{
				firstLine = false;
				sparse = line.contains(":");
			}

			String[] split = line.split(delimiter);
			if(sparse)
			{
				int[] keys = new int[split.length - 1];
				double[] values = new double[split.length - 1];

				for(int i = 1; i < split.length; i++)
				{
					String[] split2 = split[i].split(":");
					keys[i - 1] = Integer.parseInt(split2[0]);
					maxKey = Math.max(maxKey, keys[i - 1] + 1);
					values[i - 1] = Double.parseDouble(split2[1]);
				}
				this.instances.add(new SparseInstance(Double.parseDouble(split[0]), keys, values));
			}
			else
			{
				double[] values = new double[split.length - 1];
				for(int i = 1; i < split.length; i++)
					values[i - 1] = Double.parseDouble(split[i]);
				maxKey = Math.max(maxKey, split.length - 1);
				this.instances.add(new DenseInstance(Double.parseDouble(split[0]), values));
			}
			this.numValues = maxKey;
		}
		br.close();
	}

	protected int numOfLines(File file) throws IOException
	{
		BufferedReader reader = new BufferedReader(new FileReader(file));
		int lines = 0;
		while(reader.readLine() != null)
			lines++;
		reader.close();
		return lines;
	}

	/**
	 * Shuffles the internal Array List of instances according to a fixed random seed.
	 */
	public void shuffle(int seed)
	{
		java.util.Random random = new java.util.Random(seed);
		Collections.shuffle(instances, random);
	}

	public double getMaxTarget()
	{
		double[] targets = this.getTargets();
		double max = -Double.MAX_VALUE;
		for(int i = 0; i < targets.length; i++)
		{
			if(targets[i] > max)
			{
				max = targets[i];
			}
		}
		return max;
	}

	public double getMinTarget()
	{
		double[] targets = this.getTargets();
		double min = Double.MAX_VALUE;
		for(int i = 0; i < targets.length; i++)
		{
			if(targets[i] < min)
			{
				min = targets[i];
			}
		}
		return min;
	}

	/**
	 * Gets the targets of all instances.
	 * 
	 * @return double array of target values.
	 */
	public double[] getTargets()
	{
		double[] ret = new double[this.numInstances()];
		for(int i = 0; i < this.numInstances(); i++)
		{
			ret[i] = this.instance(i).target();
		}
		return ret;
	}

	/**
	 * Overwrites target values with new target values, i.e. predicted values
	 * 
	 * @param newTargets
	 */
	public void setTargets(double[] newTargets)
	{
		if(newTargets.length != this.numInstances())
		{
			System.err.println("Targets cannot be overwritten due to different length of newTargets and number of Instances!");
			return;
		}
		for(int i = 0; i < newTargets.length; i++)
		{
			this.instance(i).setTarget(newTargets[i]);
		}
	}

	/**
	 * Adds a given instance to the Instances object.
	 * 
	 * @param instance
	 * @return true if instance was successfully added.
	 */
	public boolean add(Instance instance)
	{
		int maxKey = instance.getKeys()[instance.getKeys().length - 1];
		if(maxKey > this.numValues)
			throw new IllegalArgumentException("The instance has " + maxKey + " attributes but only " + this.numValues + " are allowed.");
		return this.instances.add(instance);
	}

	/**
	 * Adds all of the given Instances to the Instances object.
	 * 
	 * @param instances
	 * @return
	 */
	public boolean addAll(Instances instances)
	{
		int maxKey = 0;
		for(int i = 0; i < instances.numInstances(); i++)
			maxKey = Math.max(maxKey, instances.instance(i).getKeys()[instances.instance(i).getKeys().length - 1]);
		if(maxKey > this.numValues)
			throw new IllegalArgumentException("The instance has " + maxKey + " attributes but only " + this.numValues + " are allowed.");
		return this.instances.addAll(instances.instances);
	}

	public boolean remove(Instance instance)
	{
		return this.instances.remove(instance);
	}

	public Instance instance(int i)
	{
		return this.instances.get(i);
	}

	public int numInstances()
	{
		return this.instances.size();
	}

	public int numValues()
	{
		return this.numValues;
	}

	public void saveToLibsvm(File file) throws IOException
	{
		BufferedWriter bw = new BufferedWriter(new OutputStreamWriter(new FileOutputStream(file), "utf-8"));
		for(Instance instance : this.instances)
		{
			bw.write("" + instance.target());
			int[] keys = instance.getKeys();
			double[] values = instance.getValues();
			for(int i = 0; i < keys.length; i++)
			{
				if(values[i] != 0)
				bw.write(" " + keys[i] + ":" + values[i]);
			}
			bw.newLine();
		}
		bw.close();
	}

	public void saveToSVMLight(File file) throws IOException
	{
		BufferedWriter bw = new BufferedWriter(new OutputStreamWriter(new FileOutputStream(file), "utf-8"));
		for(Instance instance : this.instances)
		{
			bw.write("" + instance.target());
			int[] keys = instance.getKeys();
			double[] values = instance.getValues();
			for(int i = 0; i < keys.length; i++)
			{
				if(values[i] != 0)
				bw.write(" " + (keys[i] + 1) + ":" + values[i]);
			}
			bw.newLine();
		}
		bw.close();
	}

	public void saveToDense(String filename) throws IOException
	{
		saveToDense(new File(filename));
	}
	
	public void saveToDense(String filename, String delimiter) throws IOException
	{
		saveToDense(new File(filename), delimiter);
	}

	public void saveToDense(File file) throws IOException
	{
		saveToDense(file, " ");
	}

	public void saveToDense(File file, String delimiter) throws IOException
	{
		BufferedWriter bw = new BufferedWriter(new OutputStreamWriter(new FileOutputStream(file), "utf-8"));
		for(Instance instance : this.instances)
		{
			bw.write("" + instance.target());
			int[] keys = instance.getKeys();
			double[] values = instance.getValues();
			for(int i = 0; i < keys.length; i++)
			{
				bw.write(delimiter + values[i]);
			}
			bw.newLine();
		}
		bw.close();
	}

	@Override
	public Iterator<Instance> iterator()
	{
		return this.instances.iterator();
	}

}
