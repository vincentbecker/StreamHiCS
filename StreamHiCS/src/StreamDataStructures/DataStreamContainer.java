package StreamDataStructures;

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
	 * corresponding to the given dimension and the {@link Subspace} which
	 * contains this dimension. On every dimension in the {@link Subspace}
	 * except the specified one random range selections on instances (of the
	 * specified selection size) are done, representing a conditional sample for
	 * the given dimension.
	 * 
	 * @param subspace
	 *            The {@link Subspace} which contains the given dimension and on
	 *            which the constraints are placed.
	 * 
	 * @param dimension
	 *            The dimension for which a random conditional sample should be
	 *            drawn.
	 * @param selectionSize
	 *            The number of instances that should be selected per dimension.
	 * @return A double[] containing a random conditional sample corresponding
	 *         to the given dimension.
	 */
	public abstract double[] getSlicedData(Subspace subspace, int dimension,
			int selectionSize);

}
