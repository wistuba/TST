package de.ismll.kernel;

import org.apache.commons.math3.linear.MatrixUtils;
import org.apache.commons.math3.linear.RealMatrix;
import org.apache.commons.math3.linear.RealVector;

import de.ismll.core.Instance;
import de.ismll.core.Instances;

public class SEARDKernel extends Kernel
{
	private double sigma_f, sigma_y;

	private double[] sigma_l;

	/**
	 * AdaGrad update history.
	 */
	private double sumF, sumY;

	private double[] sumL;

	// private float[][][] summandsOfL2NormOfDifference;

	// private Instances instances;

	private int length;

	public SEARDKernel(int length)
	{
		this.length = length;
		this.initialize();
	}

	@Override
	protected void initialize()
	{
		this.sigma_l = new double[this.length];
		for(int i = 0; i < this.length; i++)
		{
			sigma_l[i] = 1;
		}
		this.sumL = new double[this.length];
		for(int i = 0; i < this.length; i++)
		{
			sumL[i] = 0.001;
		}
		this.sumF = 0.001;
		this.sumY = 0.001;
		this.sigma_f = 1;
		this.sigma_y = 0.001;
	}

	@Override
	public double[][] computeKernel(Instances instances)
	{
		// if(!instances.equals(this.instances))
		// {
		// precomputeSummandsOfSquaredL2NormOfDifferences(instances);
		// }
		double[][] kArray = new double[instances.numInstances()][instances.numInstances()];
		for(int i = 0; i < instances.numInstances(); i++)
			for(int j = i; j < instances.numInstances(); j++)
			{
				kArray[i][j] = this.computeValue(instances.instance(i), instances.instance(j));
				kArray[j][i] = kArray[i][j];
			}
		return kArray;
	}

	// private void precomputeSummandsOfSquaredL2NormOfDifferences(Instances instances)
	// {
	// this.summandsOfL2NormOfDifference = new float[instances.numInstances()][][];
	// for(int i = 0; i < summandsOfL2NormOfDifference.length; i++)
	// {
	// summandsOfL2NormOfDifference[i] = new float[i + 1][instances.numValues()];
	// for(int j = 0; j < summandsOfL2NormOfDifference[i].length; j++)
	// {
	// for(int d = 0; d < instances.numValues(); d++)
	// {
	// float absDiff = (float) (instances.instance(i).getValue(d) - instances.instance(j).getValue(d));
	// summandsOfL2NormOfDifference[i][j][d] = absDiff * absDiff;
	// }
	// }
	// }
	// }

	// private double computeValue(int i, int j)
	// {
	// if(i == j)
	// {
	// return Math.pow(this.sigma_f, 2) * this.getExponentialPart(i, j) + this.sigma_y * this.sigma_y;
	// }
	// else
	// {
	// return Math.pow(this.sigma_f, 2) * this.getExponentialPart(i, j);
	// }
	// }

	@Override
	public double computeValue(Instance instance1, Instance instance2)
	{
		if(instance1 == instance2)
		{
			return Math.pow(this.sigma_f, 2) * this.getExponentialPart(instance1, instance2) + this.sigma_y * this.sigma_y;
		}
		else
		{
			return Math.pow(this.sigma_f, 2) * this.getExponentialPart(instance1, instance2);
		}
	}

	// private double getExponentialPart(int i, int j)
	// {
	// double sum = 0;
	// if(j > i)
	// {
	// for(int d = 0; d < this.summandsOfL2NormOfDifference[0][0].length; d++)
	// {
	// sum += summandsOfL2NormOfDifference[j][i][d] / this.sigma_l[d] / this.sigma_l[d];
	// }
	// }
	// else
	// {
	// for(int d = 0; d < this.summandsOfL2NormOfDifference[0][0].length; d++)
	// {
	// sum += summandsOfL2NormOfDifference[i][j][d] / this.sigma_l[d] / this.sigma_l[d];
	// }
	// }
	// for(int k = 0; k < this.sigma_l.length; k++)
	// sum +=
	// return Math.exp(-0.5 * sum);
	// }

	private double getExponentialPart(Instance instance1, Instance instance2)
	{
		int[] keys1 = instance1.getKeys();
		double[] values1 = instance1.getValues();
		int[] keys2 = instance2.getKeys();
		double[] values2 = instance2.getValues();

		double z = 0;
		int index1 = 0, index2 = 0;
		while(index1 < keys1.length || index2 < keys2.length)
		{
			if(index1 < keys1.length && (index2 >= keys2.length || keys2[index2] > keys1[index1]))
				z += Math.pow(values1[index1++], 2) / Math.pow(this.sigma_l[keys1[index1 - 1]], 2);
			else if(index2 < keys2.length && (index1 >= keys1.length || keys1[index1] > keys2[index2]))
				z += Math.pow(values2[index2++], 2) / Math.pow(this.sigma_l[keys2[index2 - 1]], 2);
			else if(index1 < keys1.length && index2 < keys2.length)
				z += Math.pow(values1[index1++] - values2[index2++], 2) / Math.pow(this.sigma_l[keys1[index1 - 1]], 2);
		}
		return Math.exp(-z / 2);
	}

	@Override
	public void updateKernelParameters(Instances train, RealMatrix K, RealVector alpha, boolean initialize)
	{
		super.updateKernelParameters(train, K, alpha, initialize);
		// if(!train.equals(this.instances) || this.summandsOfL2NormOfDifference == null)
		// precomputeSummandsOfSquaredL2NormOfDifferences(train);

		int dimension = K.getColumnDimension();

		RealMatrix aaTMinusKInverse = this.getAATMinusKInverse(alpha, K);

		double[][] exponentialParts = new double[dimension][];
		for(int i = 0; i < exponentialParts.length; i++)
		{
			exponentialParts[i] = new double[i + 1];
			for(int j = 0; j < exponentialParts[i].length; j++)
			{
				exponentialParts[i][j] = this.getExponentialPart(train.instance(i), train.instance(j));
			}
		}

		// Derivative of sigma_y
		double[] kernelDerivativeDiag = new double[dimension];
		for(int i = 0; i < kernelDerivativeDiag.length; i++)
		{
			kernelDerivativeDiag[i] = 2 * this.sigma_y;
		}
		double gradientY = this.computeDerivative(aaTMinusKInverse, MatrixUtils.createRealDiagonalMatrix(kernelDerivativeDiag));
		this.sumY += gradientY * gradientY;
		this.sigma_y += this.learnRate / Math.sqrt(this.sumY) * gradientY;

		// Derivative of sigma_f
		double[][] kernelDerivative = new double[dimension][dimension];
		for(int i = 0; i < exponentialParts.length; i++)
		{
			for(int j = 0; j < exponentialParts[i].length; j++)
			{
				kernelDerivative[i][j] = 2 * this.sigma_f * exponentialParts[i][j];
				kernelDerivative[j][i] = kernelDerivative[i][j];
			}
		}
		double gradientF = this.computeDerivative(aaTMinusKInverse, MatrixUtils.createRealMatrix(kernelDerivative));
		this.sumF += gradientF * gradientF;
		this.sigma_f += this.learnRate / Math.sqrt(this.sumF) * gradientF;

		// Derivative of all sigma_l
		for(int d = 0; d < this.sigma_l.length; d++)
		{
			kernelDerivative = new double[dimension][dimension];
			for(int i = 0; i < exponentialParts.length; i++)
			{
				for(int j = 0; j < exponentialParts[i].length; j++)
				{
					kernelDerivative[i][j] = sigma_f * sigma_f * exponentialParts[i][j] * Math.pow(train.instance(i).getValue(d) - train.instance(j).getValue(d), 2) / this.sigma_l[d]
							/ this.sigma_l[d] / this.sigma_l[d];
					kernelDerivative[j][i] = kernelDerivative[i][j];
				}
			}
			double gradientL = this.computeDerivative(aaTMinusKInverse, MatrixUtils.createRealMatrix(kernelDerivative));
			this.sumL[d] += gradientL * gradientL;
			this.sigma_l[d] += this.learnRate / Math.sqrt(this.sumL[d]) * gradientL;
		}
	}

	@Override
	public String toString()
	{
		String s = "de.ismll.kernel.SEARDKernel:\nsigma_f=" + this.sigma_f + "\nsigma_y=" + this.sigma_y + "\n";
		for(int i = 0; i < this.sigma_l.length; i++)
		{
			s += "sigma_l[" + i + "]=" + this.sigma_l[i] + "\n";
		}
		return s;
	}

	public void setSigmaF(double sigmaF)
	{
		this.sigma_f = sigmaF;
	}

	public double getSigmaF()
	{
		return this.sigma_f;
	}

	public void setSigmaL(double[] sigmaL)
	{
		this.sigma_l = sigmaL;
	}

	public double[] getSigmaL()
	{
		return this.sigma_l;
	}

	public void setSigmaY(double sigmaY)
	{
		this.sigma_y = sigmaY;
	}

	public double getSigmaY()
	{
		return this.sigma_y;
	}
}
