package de.ismll.kernel;

import org.apache.commons.math3.linear.MatrixUtils;
import org.apache.commons.math3.linear.RealMatrix;
import org.apache.commons.math3.linear.RealVector;

import de.ismll.core.Instance;
import de.ismll.core.Instances;

public class SEKernel extends Kernel
{
	/**
	 * Kernel parameter.
	 */
	private double sigma_f, sigma_l, sigma_y;

	/**
	 * AdaGrad update history.
	 */
	private double sumF, sumL, sumY;

	private float[][] squaredL2NormOfDiff;

	private Instances instances;

	public SEKernel()
	{
		this.initialize();
	}

	@Override
	protected void initialize()
	{
		this.sumF = 0.001;
		this.sumL = 0.001;
		this.sumY = 0.001;
		this.sigma_f = 1;
		this.sigma_l = 1;
		this.sigma_y = 0.001;
	}

	@Override
	public double[][] computeKernel(Instances instances)
	{
		if(!instances.equals(this.instances))
		{
			precomputeSquaredL2NormOfDifferences(instances);
		}
		double[][] kArray = new double[instances.numInstances()][instances.numInstances()];
		for(int i = 0; i < this.squaredL2NormOfDiff.length; i++)
			for(int j = 0; j < this.squaredL2NormOfDiff[i].length; j++)
			{
				kArray[i][j] = this.computeValue(i, j);
				kArray[j][i] = kArray[i][j];
			}
		return kArray;
	}

	private void precomputeSquaredL2NormOfDifferences(Instances instances)
	{
		this.squaredL2NormOfDiff = new float[instances.numInstances()][];
		for(int i = 0; i < squaredL2NormOfDiff.length; i++)
		{
			squaredL2NormOfDiff[i] = new float[i + 1];
			for(int j = 0; j < squaredL2NormOfDiff[i].length; j++)
			{
				if(i == j)
					squaredL2NormOfDiff[i][j] = 0;
				else
					squaredL2NormOfDiff[i][j] = (float) this.getSquaredL2NormOfDiff(instances.instance(i), instances.instance(j));
			}
		}
	}

	@Override
	public double computeValue(Instance instance1, Instance instance2)
	{
		if(instance1 == instance2)
		{
			return this.sigma_f * this.sigma_f * this.getExponentialPart(instance1, instance2) + this.sigma_y * this.sigma_y;
		}
		else
		{
			return this.sigma_f * this.sigma_f * this.getExponentialPart(instance1, instance2);
		}

	}

	private double computeValue(int i, int j)
	{
		if(i == j)
		{
			return this.sigma_f * this.sigma_f * 1 + this.sigma_y * this.sigma_y;
		}
		else
		{
			return this.sigma_f * this.sigma_f * Math.exp(-squaredL2NormOfDiff[i][j] / 2 / this.sigma_l / this.sigma_l);
		}

	}

	/**
	 * Computes exp(-1/(2*sigma_l^2) * ||x_1-x_2||).
	 */
	private double getExponentialPart(Instance instance1, Instance instance2)
	{
		return Math.exp(-this.getSquaredL2NormOfDiff(instance1, instance2) / 2 / this.sigma_l / this.sigma_l);
	}

	/**
	 * Computes ||x_1-x_2||.
	 */
	private double getSquaredL2NormOfDiff(Instance instance1, Instance instance2)
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
				z += Math.pow(values1[index1++], 2);
			else if(index2 < keys2.length && (index1 >= keys1.length || keys1[index1] > keys2[index2]))
				z += Math.pow(values2[index2++], 2);
			else if(index1 < keys1.length && index2 < keys2.length)
				z += Math.pow(values1[index1++] - values2[index2++], 2);
		}
		return z;
	}

	@Override
	public void updateKernelParameters(Instances train, RealMatrix K, RealVector alpha, boolean initialize)
	{
		super.updateKernelParameters(train, K, alpha, initialize);
		if(!train.equals(this.instances) || this.squaredL2NormOfDiff == null)
			precomputeSquaredL2NormOfDifferences(train);

		int dimension = K.getColumnDimension();

		RealMatrix aaTMinusKInverse = this.getAATMinusKInverse(alpha, K);
		// Derivative for sigma_y:

		double[][] exponentialParts = new double[dimension][];
		for(int i = 0; i < this.squaredL2NormOfDiff.length; i++)
		{
			exponentialParts[i] = new double[this.squaredL2NormOfDiff[i].length];
			for(int j = 0; j < this.squaredL2NormOfDiff[i].length; j++)
			{
				exponentialParts[i][j] = Math.exp(-this.squaredL2NormOfDiff[i][j] / 2 / this.sigma_l / this.sigma_l);
			}
		}
		double[] kernelDerivativeDiag = new double[dimension];
		for(int i = 0; i < kernelDerivativeDiag.length; i++)
		{
			kernelDerivativeDiag[i] = 2 * this.sigma_y;
		}
		double gradientY = this.computeDerivative(aaTMinusKInverse, MatrixUtils.createRealDiagonalMatrix(kernelDerivativeDiag));
		this.sumY += gradientY * gradientY;
		this.sigma_y += this.learnRate / Math.sqrt(this.sumY) * gradientY;
		// Derivative for sigma_f:
		double[][] kernelDerivative = new double[dimension][dimension];
		for(int i = 0; i < this.squaredL2NormOfDiff.length; i++)
		{
			for(int j = 0; j < this.squaredL2NormOfDiff[i].length; j++)
			{
				kernelDerivative[i][j] = 2 * this.sigma_f * exponentialParts[i][j];
				kernelDerivative[j][i] = kernelDerivative[i][j];
			}
		}
		double gradientF = this.computeDerivative(aaTMinusKInverse, MatrixUtils.createRealMatrix(kernelDerivative));
		this.sumF += gradientF * gradientF;
		this.sigma_f += this.learnRate / Math.sqrt(this.sumF) * gradientF;

		// Derivative for sigma_l:
		// kernelDerivative = new double[dimension][dimension];
		for(int i = 0; i < this.squaredL2NormOfDiff.length; i++)
		{
			for(int j = 0; j < this.squaredL2NormOfDiff[i].length; j++)
			{
				kernelDerivative[i][j] = sigma_f * sigma_f * exponentialParts[i][j] * squaredL2NormOfDiff[i][j] / this.sigma_l / this.sigma_l / this.sigma_l;
				kernelDerivative[j][i] = kernelDerivative[i][j];
			}
		}
		double gradientL = this.computeDerivative(aaTMinusKInverse, MatrixUtils.createRealMatrix(kernelDerivative));
		this.sumL += gradientL * gradientL;
		this.sigma_l += this.learnRate / Math.sqrt(this.sumL) * gradientL;
	}

	@Override
	public String toString()
	{
		return "de.ismll.kernel.SEKernel:\nsigma_f=" + this.sigma_f + "\nsigma_y=" + this.sigma_y + "\nsigma_l=" + this.sigma_l + "\n";
	}

	public double getSigma_f()
	{
		return this.sigma_f;
	}

	public void setSigma_f(double sigma_f)
	{
		this.sigma_f = sigma_f;
	}

	public double getSigma_l()
	{
		return this.sigma_l;
	}

	public void setSigma_l(double sigma_l)
	{
		this.sigma_l = sigma_l;
	}

	public double getSigma_y()
	{
		return this.sigma_y;
	}

	public void setSigma_y(double sigma_y)
	{
		this.sigma_y = sigma_y;
	}
}
