package de.ismll.hylap;

import java.util.ArrayList;

import de.ismll.core.Instance;
import de.ismll.core.InstanceUtils;
import de.ismll.core.Instances;
import de.ismll.hylap.acquisitionFunction.AcquisitionFunction;
import de.ismll.hylap.surrogateModel.SurrogateModel;

public class SMBO
{
	private Instances instances;

	private Instances h;

	private AcquisitionFunction acquisitionFunction;

	private SurrogateModel surrogateModel;

	private ArrayList<Instance> candidates;

	private Instance bestInstance;

	private double time;

	public SMBO(Instances instances, AcquisitionFunction acquisitionFunction, SurrogateModel surrogateModel)
	{
		this.instances = instances;
		this.acquisitionFunction = acquisitionFunction;
		this.surrogateModel = surrogateModel;
		this.h = new Instances(instances.numValues());
		this.candidates = new ArrayList<Instance>();
		for(int i = 0; i < this.instances.numInstances(); i++)
			this.candidates.add(this.instances.instance(i));
	}

	public void iterate()
	{
		Instance x = this.acquisitionFunction.getNext(this.h, this.surrogateModel, this.candidates);
		this.candidates.remove(x);
		if(this.bestInstance == null || this.bestInstance.target() < x.target())
			this.bestInstance = x;
		this.h.add(x);
		this.surrogateModel.train(this.h);
	}

	public double getBestAccuracy()
	{
		return this.bestInstance.target();
	}

	public int getBestRank()
	{
		return InstanceUtils.getRank(this.instances, this.bestInstance);
	}

	public double getTime()
	{
		return this.time;
	}
}
