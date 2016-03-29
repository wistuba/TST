package de.ismll.core.regression;

import java.util.logging.Level;

import org.apache.commons.math3.linear.ArrayRealVector;
import org.apache.commons.math3.linear.CholeskyDecomposition;
import org.apache.commons.math3.linear.MatrixUtils;
import org.apache.commons.math3.linear.NonPositiveDefiniteMatrixException;
import org.apache.commons.math3.linear.RealMatrix;
import org.apache.commons.math3.linear.RealVector;

import de.ismll.core.Instance;
import de.ismll.core.Instances;
import de.ismll.core.Logger;
import de.ismll.kernel.Kernel;
import de.ismll.kernel.SEKernel;

public class GaussianProcessRegression implements Regression
{
	private RealVector alpha;

	private RealMatrix L;

	public Instances instances;

	private Kernel kernel = new SEKernel();

	private boolean learnKernelParameters = false;

	private RealVector y;

	private int epochs = 10;

	private double jitter = 1E-8;

	@Override
	public void train(Instances instances)
	{
		this.instances = instances;

		double[] t = new double[instances.numInstances()];
		for(int i = 0; i < instances.numInstances(); i++)
			t[i] = instances.instance(i).target();
		if(this.learnKernelParameters)
		{
			// Logger.info("Learning Kernel parameters for " + this.epochs + " epochs.");
			for(int iter = 0; iter < this.epochs; iter++)
			{
				RealMatrix K = MatrixUtils.createRealMatrix(this.kernel.computeKernel(this.instances));
				CholeskyDecomposition cd = this.choleskyDecomposition(K);
				this.L = cd.getL();
				this.alpha = this.estimateAlpha();

				// To avoid computing the negative log-likelihood if not necessary.
				if(Logger.LEVEL == Level.FINE || Logger.LEVEL == Level.FINER || Logger.LEVEL == Level.FINEST)
				{
					if(iter % 10 == 9)
						Logger.fine("Negative log-likelihood in epoch " + (iter + 1) + " of " + this.epochs + ": " + this.computeNegativeLogLikelihood());
				}
				this.kernel.updateKernelParameters(instances, K, alpha, iter == 0);
			}
		}

		// Logger.info("Kernel information:\n" + this.kernel.toString());

		RealMatrix K = MatrixUtils.createRealMatrix(this.kernel.computeKernel(this.instances));

		CholeskyDecomposition cd = this.choleskyDecomposition(K);
		this.L = cd.getL();
		this.alpha = this.estimateAlpha();
	}

	private CholeskyDecomposition choleskyDecomposition(RealMatrix K)
	{
		CholeskyDecomposition cd = null;
		boolean passed = false;
		int numNotPassed = 0;
		while(!passed && numNotPassed < 10)
		{
			try
			{
				for(int i = 0; i < K.getRowDimension(); i++)
					K.setEntry(i, i, K.getEntry(i, i) + this.jitter);
				cd = new CholeskyDecomposition(K);
				passed = true;
			}
			catch(NonPositiveDefiniteMatrixException e)
			{
				this.jitter *= 1.01;
			}
			numNotPassed++;
		}
		if(!passed)
		{
			throw new IllegalArgumentException("Adding Jitter did not work.");
		}
		this.jitter = 1E-8;
		return cd;
	}

	private RealVector estimateAlpha()
	{
		double[] yArray = new double[instances.numInstances()];
		for(int i = 0; i < instances.numInstances(); i++)
			yArray[i] = instances.instance(i).target();

		RealVector alpha = new ArrayRealVector(yArray);
		this.y = new ArrayRealVector(yArray);
		MatrixUtils.solveLowerTriangularSystem(this.L, alpha);
		MatrixUtils.solveUpperTriangularSystem(this.L.transpose(), alpha);
		return alpha;
	}

	private double computeNegativeLogLikelihood()
	{
		double logDeterminant = 0;
		for(int i = 0; i < this.L.getColumnDimension(); i++)
			logDeterminant += Math.log(this.L.getEntry(i, i)) / Math.log(Math.E);
		return -0.5 * this.y.dotProduct(alpha) - logDeterminant - this.L.getColumnDimension() / 2 * Math.log(2 * Math.PI) / Math.log(Math.E);
	}

	@Override
	public double predict(Instance instance)
	{
		return this.getKStar(instance).dotProduct(this.alpha);
	}

	public double[] predictWithUncertainty(Instance instance)
	{
		if(this.instances == null)
			return new double[] { 0, Double.POSITIVE_INFINITY };
		double[] pred = new double[2];
		ArrayRealVector kStar = this.getKStar(instance);
		pred[0] = kStar.dotProduct(this.alpha);
		MatrixUtils.solveLowerTriangularSystem(this.L, kStar);
		pred[1] = Math.sqrt(-kStar.dotProduct(kStar) + this.kernelFunction(instance, instance));
		return pred;
	}

	public void onlineUpdate(Instance instance)
	{
		if(this.instances == null)
			throw new IllegalArgumentException("Model was not trained before so it cannot be updated");
		RealVector l = this.getKStar(instance);
		MatrixUtils.solveLowerTriangularSystem(this.L, l);
		double lStar = Math.sqrt(this.kernelFunction(instance, instance) - l.dotProduct(l));
		RealMatrix L_new = MatrixUtils.createRealMatrix(this.L.getRowDimension() + 1, this.L.getColumnDimension() + 1);
		L_new.setSubMatrix(this.L.getData(), 0, 0);
		L_new.setEntry(L_new.getRowDimension() - 1, L_new.getColumnDimension() - 1, lStar);
		L_new.setSubMatrix(new double[][] { l.toArray() }, L_new.getRowDimension() - 1, 0);
		this.L = L_new;
		this.instances.add(instance);
		this.alpha = this.estimateAlpha();
	}

	private ArrayRealVector getKStar(Instance instance)
	{
		double[] kStarArray = new double[this.instances.numInstances()];
		for(int i = 0; i < this.instances.numInstances(); i++)
		{
			kStarArray[i] = this.kernelFunction(this.instances.instance(i), instance);
		}
		return new ArrayRealVector(kStarArray, false);
	}

	private double kernelFunction(Instance inst1, Instance inst2)
	{
		return this.kernel.computeValue(inst1, inst2);
	}

	public Kernel getKernel()
	{
		return this.kernel;
	}

	public void setKernel(Kernel kernel)
	{
		this.kernel = kernel;
	}

	public boolean isLearnKernelParameters()
	{
		return this.learnKernelParameters;
	}

	public void setLearnKernelParameters(boolean learnKernelParameters)
	{
		this.learnKernelParameters = learnKernelParameters;
	}

	public int getEpochs()
	{
		return this.epochs;
	}

	public void setEpochs(int epochs)
	{
		this.epochs = epochs;
	}

	@Override
	public double[] predict(Instances instances)
	{
		double[] ret = new double[instances.numInstances()];
		for(int i = 0; i < instances.numInstances(); i++)
		{
			ret[i] = this.predict(instances.instance(i));
		}
		return ret;
	}

	public double[] getAlpha()
	{
		return this.alpha.toArray();
	}
	
	public void updateLabels(double[] labels) {
		if (this.instances.numInstances() != labels.length) {
			System.err.println("Labels cannot be updated as the number of instances and labels differs");
			return;
		}
		int i = 0;
		for (Instance instance : this.instances) {
			instance.setTarget(labels[i]);
			i++;
		}
		
		
		
	}
	
}
