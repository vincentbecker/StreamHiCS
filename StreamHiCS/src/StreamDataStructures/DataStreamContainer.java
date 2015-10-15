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
	 * The number of instances currently contained in the
	 * {@link DataStreamContainer}.
	 */
	public int numberOfInstances;

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
	public double[] getProjectedData(int dimension) {
		// Selection alpha does not matter here
		Selection selectedIndexes = new Selection(numberOfInstances, 1);
		// Fill the list with all the indexes to select all data
		selectedIndexes.fillRange();
		return getSelectedData(dimension, selectedIndexes);
	}

	/**
	 * Returns the one dimensional data of a random conditional sample
	 * corresponding to the last dimension in the int[] and the {@link Subspace}
	 * which contains this dimension. On every dimension in the {@link Subspace}
	 * except the specified one random range selections on instances (of the
	 * specified selection size) are done, representing a conditional sample for
	 * the given dimension.
	 * 
	 * @param shuffledDimensions
	 *            The dimensions. The last one is the one for which a random
	 *            conditional sample should be drawn.
	 * @param selectionAlpha
	 *            The fraction of instances that should be selected per
	 *            dimension (i.e. the number of selected instances becomes
	 *            smaller per selection step).
	 * @return A double[] containing a random conditional sample corresponding
	 *         to the given dimension.
	 */
	public double[] getSlicedData(int[] shuffledDimensions, double selectionAlpha) {
		double[] dimData;
		Selection selectedIndexes = new Selection(numberOfInstances, selectionAlpha);
		// Fill the list with all the indexes
		selectedIndexes.fillRange();

		for (int i = 0; i < shuffledDimensions.length - 1; i++) {
			// Get all the data for the specific dimension that is selected
			dimData = getSelectedData(shuffledDimensions[i], selectedIndexes);
			// Reduce the number of indexes according to a new selection in
			// the current dimension
			selectedIndexes.select(dimData);
		}

		// Get the selected data from the last dimension
		return getSelectedData(shuffledDimensions[shuffledDimensions.length - 1], selectedIndexes);
	}

	/**
	 * Returns the data stored in this {@link DataStreamContainer} corresponding
	 * to the given dimension and the specified indexes.
	 * 
	 * @param dimension
	 *            The dimension the data is taken from
	 * @param selectedIndexes
	 *            The indexes of the data point which are selected.
	 * @return The data stored in this {@link DataStreamContainer} corresponding
	 *         to the given dimension and the specified indexes.
	 */
	public abstract double[] getSelectedData(int dimension, Selection selectedIndexes);
}
