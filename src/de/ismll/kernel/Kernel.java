package de.ismll.kernel;

import org.apache.commons.math3.linear.LUDecomposition;
import org.apache.commons.math3.linear.RealMatrix;
import org.apache.commons.math3.linear.RealVector;

import de.ismll.core.Instance;
import de.ismll.core.Instances;

public abstract class Kernel
{
	protected double learnRate = 0.1;

	public abstract double computeValue(Instance instance1, Instance instance2);

	public abstract double[][] computeKernel(Instances instances);

	/**
	 * One update step of the kernel parameters. Call a couple of times to iteratively maximize the likelihood on train. Do not forget to recompute K and alpha.
	 * 
	 * @param initialize
	 *            If it is true, reinitialize (setting history of adagrad to zero etc.)
	 */
	public void updateKernelParameters(Instances train, RealMatrix K, RealVector alpha, boolean initialize)
	{
		if(initialize)
			this.initialize();
	}

	protected abstract void initialize();

	protected double computeDerivative(RealMatrix aaTMinusKInverse, RealMatrix kernelDerivativeMatrix)
	{
		double trace = 0;
		for(int i = 0; i < aaTMinusKInverse.getRowDimension(); i++)
		{
			for(int j = 0; j < aaTMinusKInverse.getColumnDimension(); j++)
			{
				trace += aaTMinusKInverse.getEntry(i, j) * kernelDerivativeMatrix.getEntry(j, i);
			}
		}
		return 0.5 * trace;
	}

	protected RealMatrix getAATMinusKInverse(RealVector alpha, RealMatrix K)
	{
		return alpha.outerProduct(alpha).add(new LUDecomposition(K).getSolver().getInverse().scalarMultiply(-1));
	}
}
