package de.ismll.hylap.surrogateModel;

import de.ismll.core.Instance;
import de.ismll.core.Instances;

public interface SurrogateModel
{
	public void train(Instances instances);

	/**
	 * Returns an array where the first entry is the mean and the second the standard deviation.
	 * 
	 * @param instance
	 */
	public double[] predict(Instance instance);
}
