package de.ismll.core.regression;

import de.ismll.core.Instance;
import de.ismll.core.Instances;

public interface Regression
{
	
	public void train(Instances instances);
	
	public double predict(Instance instance);
	
	public double[] predict(Instances instances);
	
}
