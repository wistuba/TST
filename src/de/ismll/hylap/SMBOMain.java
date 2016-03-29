package de.ismll.hylap;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;

import org.apache.commons.math3.stat.descriptive.moment.Mean;
import org.apache.commons.math3.stat.descriptive.moment.StandardDeviation;

import de.ismll.core.Instance;
import de.ismll.core.Instances;
import de.ismll.core.Logger;
import de.ismll.core.Random;
import de.ismll.core.regression.GaussianProcessRegression;
import de.ismll.hylap.acquisitionFunction.AcquisitionFunction;
import de.ismll.hylap.acquisitionFunction.ExpectedImprovement;
import de.ismll.hylap.surrogateModel.SurrogateModel;
import de.ismll.hylap.surrogateModel.TwoStageSurrogate;

public class SMBOMain
{
	public static void help()
	{
		System.out
				.println("============= Mandatory Parameters =============\n"
						+ "-f\t\tPath to the folder where your datasets are stored.\n"
						+ "-dataset\tName of the dataset to evaluate.\n"
						+ "-tries\t\tNumber of steps for the SMBO algorithm.\n"
						+ "-s\t\tThe surrogate model. \"tst-m\" (TST with meta-features), \"tst-r\" (TST with pairwise comparisons)\n"
						+ "-bandwidth\tBandwidth (TST-M: SVM: 3.25; TST-R: SVM: 0.1, Weka: 0.9)\n"
						+ "-hpRange\tNumber of hyperparameters (SVM: 6, Weka: 103)\n"
						+ "-hpIndicatorRange\tNumber of hyperparameters that are indicators (SVM: 3, Weka: 64)\n"
						+ "\n============= Optional Parameters =============\n"
						+ "-seed\t\tRandom seed (Default: 0, Random: r)\n"
						+ "-iter\t\tNumber of iterations, results are averaged.\n"
						+ "-sparseGrid\tWhether to use only a subgrid of the data for training (default: true)"
						);
		System.exit(0);
	}

	public static void main(String[] args) throws IOException
	{
		HashMap<String, String> argsMap = new HashMap<String, String>();
		for(int i = 0; i < args.length; i++)
			argsMap.put(args[i], args[++i]);

		// Mandatory parameters
		if(!argsMap.containsKey("-dataset") || !argsMap.containsKey("-s") || !argsMap.containsKey("-f") || !argsMap.containsKey("-tries")
				|| !argsMap.containsKey("-hpRange") || !argsMap.containsKey("-bandwidth") || !argsMap.containsKey("-hpIndicatorRange"))
			help();
		String datasetName = argsMap.get("-dataset");
		String dataFolder = argsMap.get("-f");
		File[] files = new File(dataFolder).listFiles();
		int maxTries = Integer.parseInt(argsMap.get("-tries"));
		int numIters = 1;
		if(argsMap.containsKey("-iter"))
		{
			numIters = Integer.parseInt(argsMap.get("-iter"));
		}

		if(!new File(dataFolder + "/" + datasetName).exists())
		{
			Logger.severe("Data set " + datasetName + " does not exist in folder " + new File(dataFolder).getAbsolutePath() + ".");
			System.exit(1);
		}
		// Set seed
		if(argsMap.containsKey("-seed"))
		{
			if(argsMap.get("-seed").equals("r"))
				Random.setSeed(System.currentTimeMillis());
			else
				Random.setSeed(Long.parseLong(argsMap.get("-seed")));
		}
		else
			Random.setSeed(0);

		if(argsMap.containsKey("-hpRange"))
			HyperparameterCombination.HYPERPARAMETER_INDEX_RANGE_MAX = Integer.parseInt(argsMap.get("-hpRange"));
		if(argsMap.containsKey("-hpIndicatorRange"))
			HyperparameterCombination.HYPERPARAMETER_INDICATOR_RANGE_MAX = Integer.parseInt(argsMap.get("-hpIndicatorRange"));

		double bandwidth = Double.parseDouble(argsMap.get("-bandwidth"));
		boolean sparseGrid = true;
		if(argsMap.containsKey("-sparseGrid"))
			sparseGrid = Boolean.parseBoolean(argsMap.get("-sparseGrid"));

		Logger.info("Loading data sets from " + new File(dataFolder).getAbsolutePath() + ".");
		Instances[] train = new Instances[files.length - 1];
		int testId = -1;

		for(int i = 0; i < files.length; i++)
		{
			if(files[i].getName().equals(datasetName))
				testId = i;
		}

		int l = 0;
		for(int j = 0; j < files.length; j++)
		{
			if(j != testId)
			{
				train[l++] = new Instances(files[j]);
			}
		}

		if(sparseGrid)
		{
			// Remove instances from the grid
			HashMap<Double, Integer>[] hpIndex = new HashMap[HyperparameterCombination.HYPERPARAMETER_INDEX_RANGE_MAX - HyperparameterCombination.HYPERPARAMETER_INDICATOR_RANGE_MAX];
			for(int i = 0; i < hpIndex.length; i++)
			{
				hpIndex[i] = new HashMap<Double, Integer>();
				ArrayList<Double> hyperparameterValues = new ArrayList<Double>();
				for(int j = 0; j < train[0].numInstances(); j++)
				{
					if(!hyperparameterValues.contains(train[0].instance(j).getValue(HyperparameterCombination.HYPERPARAMETER_INDICATOR_RANGE_MAX + i)))
						hyperparameterValues.add(train[0].instance(j).getValue(HyperparameterCombination.HYPERPARAMETER_INDICATOR_RANGE_MAX + i));
				}
				Collections.sort(hyperparameterValues);
				for(int j = 0; j < hyperparameterValues.size(); j++)
				{
					hpIndex[i].put(hyperparameterValues.get(j), j);
				}
			}
			for(int i = 0; i < train.length; i++)
			{
				ArrayList<Instance> instancesToBeRemoved = new ArrayList<Instance>();
				for(int j = 0; j < train[i].numInstances(); j++)
				{
					Instance instance = train[i].instance(j);
					for(int h = 0; h < hpIndex.length; h++)
					{
						double instanceValue = instance.getValue(HyperparameterCombination.HYPERPARAMETER_INDICATOR_RANGE_MAX + h);
						if((hpIndex[h].get(instanceValue) + 2) % 3 != 0)
						{
							if(HyperparameterCombination.HYPERPARAMETER_INDICATOR_RANGE_MAX == 0 || HyperparameterCombination.HYPERPARAMETER_INDICATOR_RANGE_MAX != 0 && instanceValue != 0)
							{
								instancesToBeRemoved.add(instance);
								break;
							}
						}
					}
				}
				for(Instance inst : instancesToBeRemoved)
					train[i].remove(inst);
			}
		}

		Instances testData = new Instances(files[testId]);

		Logger.info("Starting the SMBO framework.");
		double[][] acc = new double[maxTries][numIters];
		double[][] rank = new double[maxTries][numIters];
		double[] time = new double[maxTries];
		int[] count = new int[maxTries];
		GaussianProcessRegression[] tstSurrogates = null;
		HashMap<Instance,Double>[] tstCachedPredictions = null;
		for(int iter = 0; iter < numIters; iter++)
		{
			Logger.info("Starting iteration " + (iter + 1) + ".");
			AcquisitionFunction a = new ExpectedImprovement();

			SurrogateModel s = null;
			if(argsMap.get("-s").equals("tst-m") || argsMap.get("-s").equals("tst-r"))
			{
				if(bandwidth <= 0)
				{
					Logger.severe("Bandwidth not set or not positive.");
					System.exit(1);
				}
				if(tstSurrogates == null)
				{
					TwoStageSurrogate kr = new TwoStageSurrogate(train, testData, bandwidth, argsMap.get("-s").equals("tst-m"));
					tstSurrogates = kr.getSurrogates();
					tstCachedPredictions = kr.getCachedPredictions();
					s = kr;
				}
				else
					s = new TwoStageSurrogate(train, testData, tstSurrogates, tstCachedPredictions, bandwidth, argsMap.get("-s").equals("tst-m"));
			}
			else
			{
				Logger.severe("Unknown surrogate function \"" + argsMap.get("-s") + "\"");
				System.exit(1);
			}

			long start = System.nanoTime();
			SMBO smbo = new SMBO(testData, a, s);
			for(int j = 0; j < maxTries; j++)
			{
				if(j > 0 && rank[j - 1][iter] == 1)
				{
					acc[j][iter] = acc[j - 1][iter];
					rank[j][iter] = rank[j - 1][iter];
				}
				else
				{
					smbo.iterate();
					acc[j][iter] = smbo.getBestAccuracy();
					rank[j][iter] = smbo.getBestRank();
					time[j] += (double) (System.nanoTime() - start) / 1000000;
					count[j]++;
				}
			}
		}

		StandardDeviation sd = new StandardDeviation();
		Mean mean = new Mean();

		Logger.info("Printing results to console.");
		System.out.println("Accuracy(mean),Accuracy(sd),Rank(mean),Rank(sd),Time in ms");
		for(int j = 0; j < maxTries; j++)
		{
			System.out.println(mean.evaluate(acc[j]) + "," + sd.evaluate(acc[j]) + "," + mean.evaluate(rank[j]) + "," + sd.evaluate(rank[j]) + "," + (time[j] / count[j]));
		}
	}
}
