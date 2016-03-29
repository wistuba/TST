package de.ismll.core;

public abstract class Random
{
	private static java.util.Random random = new java.util.Random(0);

	public static java.util.Random getInstance()
	{
		return random;
	}
	
	/**
	 * Returns the next pseudorandom, uniformly distributed double value between 0.0 and 1.0 from this random number generator's sequence.
	 */
	public static double nextDouble()
	{
		return random.nextDouble();
	}

	/**
	 * Returns a pseudorandom, uniformly distributed int value between 0 (inclusive) and the specified value (exclusive), drawn from this random number generator's sequence.
	 */
	public static int nextInt(int n)
	{
		return random.nextInt(n);
	}

	/**
	 * Returns the next pseudorandom, Gaussian ("normally") distributed double value with mean 0.0 and standard deviation 1.0 from this random number generator's sequence.
	 */
	public static double nextGaussian()
	{
		return random.nextGaussian();
	}

	/**
	 * Returns the next pseudorandom, Gaussian distributed double value with specified mean and standard deviation from this random number generator's sequence.
	 */
	public static double nextGaussian(double mean, double sd)
	{
		return random.nextGaussian() * sd + mean;
	}

	/**
	 * Sets the seed of this random number generator using a single long seed.
	 */
	public static void setSeed(long seed)
	{
		random.setSeed(seed);
	}
}
