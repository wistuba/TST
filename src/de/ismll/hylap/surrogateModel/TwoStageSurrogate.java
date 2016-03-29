package de.ismll.hylap.surrogateModel;

import java.util.Arrays;
import java.util.HashMap;

import de.ismll.core.Instance;
import de.ismll.core.InstanceUtils;
import de.ismll.core.Instances;
import de.ismll.core.regression.GaussianProcessRegression;
import de.ismll.hylap.HyperparameterCombination;
import de.ismll.kernel.SEARDKernel;

public class TwoStageSurrogate implements SurrogateModel
{
	private GaussianProcessRegression model = new GaussianProcessRegression();

	private GaussianProcessRegression[] gp;

	private Instances data;

	private int epochs = 100;

	private Kernel kernel;

	private double bandwidth;

	private double[] similarity;

	private HashMap<Instance, Double>[] cachedPredictions;

	private Instances[] untouchedTrain;

	private Instances untouchedKnownTest;

	/**
	 * Needed for the EuclideanDistance Kernel to compute the distance based on meta-features. Very lazy way of doing it.
	 */
	private Instance metafeatureInstance;

	public TwoStageSurrogate(Instances[] train, Instances test, double bandwidth, boolean metafeatures)
	{
		this.untouchedTrain = train;
		this.bandwidth = bandwidth;
		Instances[] scaledTrain = new Instances[train.length];
		for(int i = 0; i < train.length; i++)
		{
			scaledTrain[i] = new Instances(HyperparameterCombination.HYPERPARAMETER_INDEX_RANGE_MAX);
			double max = Double.NEGATIVE_INFINITY;
			double min = Double.POSITIVE_INFINITY;
			for(int j = 0; j < train[i].numInstances(); j++)
			{
				double target = train[i].instance(j).target();
				max = Math.max(target, max);
				min = Math.min(target, min);
			}

			for(int j = 0; j < train[i].numInstances(); j++)
			{
				Instance instance = train[i].instance(j);
				scaledTrain[i].add(InstanceUtils.createDenseInstance((instance.target() - min) / ((max - min) == 0 ? 1 : (max - min)),
						Arrays.copyOfRange(instance.getValues(), 0, HyperparameterCombination.HYPERPARAMETER_INDEX_RANGE_MAX)));
			}
		}

		this.gp = new GaussianProcessRegression[scaledTrain.length];
		for(int d = 0; d < scaledTrain.length; d++)
		{
			this.gp[d] = new GaussianProcessRegression();
			this.gp[d].setKernel(new SEARDKernel(HyperparameterCombination.HYPERPARAMETER_INDEX_RANGE_MAX));
			this.gp[d].setLearnKernelParameters(true);
			this.gp[d].setEpochs(100);
			this.gp[d].train(scaledTrain[d]);
		}

		// Precompute similarities for speed-up
		this.cachedPredictions = new HashMap[this.gp.length];
		for(int d = 0; d < scaledTrain.length; d++)
		{
			this.cachedPredictions[d] = new HashMap<Instance, Double>();
			for(int i = 0; i < test.numInstances(); i++)
			{
				Instance instance = test.instance(i);
				Instance scaledInstance = InstanceUtils.createDenseInstance(instance.target(), Arrays.copyOfRange(instance.getValues(), 0, HyperparameterCombination.HYPERPARAMETER_INDEX_RANGE_MAX));
				this.cachedPredictions[d].put(instance, this.gp[d].predict(scaledInstance));
			}
		}

		if(metafeatures)
		{
			this.kernel = new EuclideanDistance();
			this.metafeatureInstance = test.instance(0);
		}
		else
			this.kernel = new KendallTauCorrelation();

		this.similarity = new double[this.gp.length];
		for(int i = 0; i < this.gp.length; i++)
		{
			this.similarity[i] = this.kernel.selfSimilarity();
		}
	}

	public TwoStageSurrogate(Instances[] train, Instances test, GaussianProcessRegression[] surrogates, HashMap<Instance, Double>[] cachedPredictions, double bandwidth, boolean metafeatures)
	{
		this.untouchedTrain = train;
		this.bandwidth = bandwidth;
		Instances[] scaledTrain = new Instances[train.length];
		for(int i = 0; i < train.length; i++)
		{
			scaledTrain[i] = new Instances(HyperparameterCombination.HYPERPARAMETER_INDEX_RANGE_MAX);
			double max = Double.NEGATIVE_INFINITY;
			double min = Double.POSITIVE_INFINITY;
			for(int j = 0; j < train[i].numInstances(); j++)
			{
				double target = train[i].instance(j).target();
				max = Math.max(target, max);
				min = Math.min(target, min);
			}

			for(int j = 0; j < train[i].numInstances(); j++)
			{
				Instance instance = train[i].instance(j);
				scaledTrain[i].add(InstanceUtils.createDenseInstance((instance.target() - min) / ((max - min) == 0 ? 1 : (max - min)),
						Arrays.copyOfRange(instance.getValues(), 0, HyperparameterCombination.HYPERPARAMETER_INDEX_RANGE_MAX)));
			}
		}

		this.gp = surrogates;
		this.cachedPredictions = cachedPredictions;

		if(metafeatures)
		{
			this.kernel = new EuclideanDistance();
			this.metafeatureInstance = test.instance(0);
		}
		else
			this.kernel = new KendallTauCorrelation();

		this.similarity = new double[this.gp.length];
		for(int i = 0; i < this.gp.length; i++)
		{
			this.similarity[i] = this.kernel.selfSimilarity();
		}
	}

	public GaussianProcessRegression[] getSurrogates()
	{
		return this.gp;
	}

	public HashMap<Instance, Double>[] getCachedPredictions()
	{
		return this.cachedPredictions;
	}

	@Override
	public void train(Instances instances)
	{
		this.untouchedKnownTest = instances;
		this.data = new Instances(HyperparameterCombination.HYPERPARAMETER_INDEX_RANGE_MAX);
		for(int i = 0; i < instances.numInstances(); i++)
		{
			Instance instance = instances.instance(i);
			this.data.add(InstanceUtils.createDenseInstance(instance.target(), Arrays.copyOfRange(instance.getValues(), 0, HyperparameterCombination.HYPERPARAMETER_INDEX_RANGE_MAX)));
		}
		this.model = new GaussianProcessRegression();
		this.model.setKernel(new SEARDKernel(this.data.numValues()));
		this.model.setLearnKernelParameters(true);
		this.model.setEpochs(this.epochs);
		this.epochs -= 5;
		this.epochs = Math.max(this.epochs, 0);
		this.model.train(this.data);

		for(int i = 0; i < this.gp.length; i++)
		{
			this.similarity[i] = this.kernel.kernel(i);
		}
	}

	@Override
	public double[] predict(Instance instance)
	{
		Instance scaledInstance = InstanceUtils.createDenseInstance(instance.target(), Arrays.copyOfRange(instance.getValues(), 0, HyperparameterCombination.HYPERPARAMETER_INDEX_RANGE_MAX));
		double[] prediction = this.model.predictWithUncertainty(scaledInstance);
		double denominator = this.kernel.selfSimilarity();
		for(int i = 0; i < this.gp.length; i++)
		{
			double similarity = this.similarity[i];
			prediction[0] += this.cachedPredictions[i].get(instance) * similarity;
			denominator += similarity;
		}
		prediction[0] /= denominator;

		// If the standard deviation is infinity, the mean does not matter.
		if(prediction[1] == Double.POSITIVE_INFINITY)
			prediction[1] = 1000;

		return prediction;
	}

	private interface Kernel
	{
		public double kernel(int index);

		public double selfSimilarity();
	}

	private class KendallTauCorrelation implements Kernel
	{
		@Override
		public double kernel(int index)
		{
			if(untouchedKnownTest == null || untouchedKnownTest.numInstances() < 2)
				return this.selfSimilarity();
			double discordantPairs = 0;
			int totalPairs = 0;
			for(int i = 0; i < untouchedKnownTest.numInstances(); i++)
			{
				Instance instance1 = untouchedKnownTest.instance(i);
				for(int j = i + 1; j < untouchedKnownTest.numInstances(); j++)
				{
					Instance instance2 = untouchedKnownTest.instance(j);
					if(instance1.target() < instance2.target() ^ cachedPredictions[index].get(instance1) < cachedPredictions[index].get(instance2))
					{
						discordantPairs++;
					}
					totalPairs++;
				}
			}

			double t = discordantPairs / totalPairs / bandwidth;
			return(t < 1 ? 0.75 * (1 - t * t) : 0);
		}

		@Override
		public double selfSimilarity()
		{
			return 0.75;
		}
	}

	private class EuclideanDistance implements Kernel
	{
		@Override
		public double kernel(int index)
		{
			Instance instance1 = metafeatureInstance;
			Instance instance2 = untouchedTrain[index].instance(0);

			double result = 0;
			int index1 = 0, index2 = 0;
			int[] keys1 = instance1.getKeys(), keys2 = instance2.getKeys();
			double[] values1 = instance1.getValues(), values2 = instance2.getValues();
			while(index1 < keys1.length && index2 < keys2.length)
			{
				if(keys1[index1] < HyperparameterCombination.HYPERPARAMETER_INDEX_RANGE_MAX)
					index1++;
				else if(keys2[index2] < HyperparameterCombination.HYPERPARAMETER_INDEX_RANGE_MAX)
					index2++;
				else
				{
					if(keys1[index1] < keys2[index2])
						result += Math.pow(values1[index1++], 2);
					else if(keys1[index1] > keys2[index2])
						result += Math.pow(values2[index2++], 2);
					else
						result += Math.pow(values1[index1++] - values2[index2++], 2);
				}
			}
			double dist = Math.sqrt(result);
			double t = dist / bandwidth;
			return(t < 1 ? 0.75 * (1 - t * t) : 0);
		}

		@Override
		public double selfSimilarity()
		{
			return 0.75;
		}
	}
}
