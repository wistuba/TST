package de.ismll.hylap.acquisitionFunction;

import java.util.ArrayList;

import org.apache.commons.math3.distribution.NormalDistribution;

import de.ismll.core.Instance;
import de.ismll.core.Instances;
import de.ismll.core.Random;
import de.ismll.hylap.surrogateModel.SurrogateModel;

public class ExpectedImprovement implements AcquisitionFunction
{
	private double xi = 0.01;

	private static final NormalDistribution ndist = new NormalDistribution();

	@Override
	public Instance getNext(Instances h, SurrogateModel surrogateModel, ArrayList<Instance> candidates)
	{
		ArrayList<Instance> bestCandidates = new ArrayList<Instance>();
		double bestEI = -1;
		double yMax = -1;

		for(int i = 0; i < h.numInstances(); i++)
			yMax = Math.max(h.instance(i).target(), yMax);

		for(Instance c : candidates)
		{
			double[] yHat = surrogateModel.predict(c);
			double ei = this.getEI(yHat[0], yHat[1], yMax);

			if(bestEI < ei)
			{
				bestEI = ei;
				bestCandidates.clear();
				bestCandidates.add(c);
			}
			else if(bestEI == ei)
				bestCandidates.add(c);
		}
		return bestCandidates.get((Random.nextInt(bestCandidates.size())));
	}
	
	public double getEI(double mu, double sigma, double yMax)
	{
		double ei = 0;
		if(sigma > 0)
		{
			double Z = (mu - xi - yMax) / sigma;
			ei = (mu - xi - yMax) * ndist.cumulativeProbability(Z) + sigma * ndist.density(Z);
		}
		return ei;
	}

	public double getXi()
	{
		return this.xi;
	}

	public void setXi(double xi)
	{
		this.xi = xi;
	}
}
