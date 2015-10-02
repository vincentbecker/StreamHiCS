package streamDataStructures;

import weka.core.Instance;

/**
 * An abstract superclass for containers holding a limited or aggregated amount
 * of the stream data.
 * 
 * @author Vincent
 *
 */
public abstract class DataStreamContainer {

	/**
	 * Adds the given {@link Instance} to the container.
	 * 
	 * @param instance
	 *            The instance to be added.
	 */
	public abstract void add(Instance instance);

	/**
	 * Clears all stored data.
	 */
	public abstract void clear();

	/**
	 * Returns the number of {@link Instance}s currently contained.
	 * 
	 * @return The number of {@link Instance}s currently contained.
	 */
	public abstract int getNumberOfInstances();

	/**
	 * Returns the one dimensional data from all the {@link Instance}s in the
	 * container concerning the given dimension.
	 * 
	 * @param dimension
	 *            The dimension of which the data is gathered.
	 * @return A double[] containing all the data held in the container
	 *         corresponding to the given dimension.
	 */
	public abstract double[] getProjectedData(int dimension);

	/**
	 * Returns the one dimensional data of a random conditional sample
	 * corresponding to the last dimension in the int[] and the {@link Subspace} which
	 * contains this dimension. On every dimension in the {@link Subspace}
	 * except the specified one random range selections on instances (of the
	 * specified selection size) are done, representing a conditional sample for
	 * the given dimension.
	 * 
	 * @param dimensions
	 *            The dimensions. The last one is the one for which a random conditional sample should be
	 *            drawn.
	 * @param selectionAlpha
	 *            The fraction of instances that should be selected per
	 *            dimension (i.e. the number of selected instances becomes
	 *            smaller per selection step).
	 * @return A double[] containing a random conditional sample corresponding
	 *         to the given dimension.
	 */
	public abstract double[] getSlicedData(int[] dimensions, double selectionAlpha);
}
