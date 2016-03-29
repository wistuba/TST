package de.ismll.hylap.acquisitionFunction;

import java.util.ArrayList;

import de.ismll.core.Instance;
import de.ismll.core.Instances;
import de.ismll.hylap.surrogateModel.SurrogateModel;

public interface AcquisitionFunction
{
	public Instance getNext(Instances h, SurrogateModel surrogateModel, ArrayList<Instance> candidates);
}
