package streamdatastructures;

import java.util.BitSet;
import subspace.Subspace;
import weka.core.Instance;

/**
 * This class represents an adapter to an underlying summarisation structure and
 * holds the necessary methods.
 * 
 * @author Vincent
 *
 */
public abstract class SummarisationAdapter {

	/**
	 * Flag to set which method to use to create the slice.
	 */
	private boolean fast = false;

	/**
	 * An array of {@link DataBundle}s, one for each dimension.
	 */
	protected DataBundle[] data;

	/**
	 * Adds an {@link Instance} to the underlying data summarisation structure.
	 * 
	 * @param instance
	 *            The instance to be added.
	 */
	public void add(Instance instance) {
		addImpl(instance);
		data = null;
	}

	/**
	 * This method is implemented by the subclass to add an {@link Instance}.
	 * 
	 * @param instance
	 *            The {@link Instance} to be added.
	 */
	public abstract void addImpl(Instance instance);

	/**
	 * Clears the underlying stream summarisation structure.
	 */
	public void clear() {
		clearImpl();
		data = null;
	}

	/**
	 * This method is implemented by the subclass to clear the underlying
	 * summarisation structure.
	 */
	public abstract void clearImpl();

	/**
	 * Returns all the data and weights from the underlying stream summarisation
	 * structure.
	 * 
	 * @return Data and weights from the stream summarisation structure in form
	 *         of an {@link ArrayList} of {@link DataBundle}s.
	 */
	public abstract DataBundle[] getData();

	/**
	 * Returns the number of elements in the stream summarisation structure.
	 * 
	 * @return The number of elements in the stream summarisation structure.
	 */
	public abstract int getNumberOfElements();

	/**
	 * Returns the data contained projected to the given reference dimension.
	 * 
	 * @param referenceDimension
	 *            The dimension the data is projected to
	 * @return The data projected to teh reference dimension.
	 */
	public DataBundle getProjectedData(int referenceDimension) {
		if (data == null) {
			getAndSortData();
		}

		int n = getNumberOfElements();
		if (n == 0) {
			return new DataBundle(new double[0], new double[0]);
		}

		// Copying the dimension data
		double[] dimData = data[referenceDimension].getData();
		double[] dimWeights = data[referenceDimension].getWeights();
		double[] dataCopy = new double[n];
		double[] weightsCopy = new double[n];
		for (int i = 0; i < n; i++) {
			dataCopy[i] = dimData[i];
			weightsCopy[i] = dimWeights[i];
		}

		return new DataBundle(dataCopy, weightsCopy);
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
	 * @return A {@link DataBundle} containing the random conditional sample
	 *         corresponding to the given dimension.
	 */
	public DataBundle getSlicedData(int[] shuffledDimensions, double selectionAlpha) {
		if (data == null) {
			getAndSortData();
		}

		int n = getNumberOfElements();
		if (n == 0) {
			return new DataBundle(new double[0], new double[0]);
		}

		double[] dimData;
		double[] weights;

		if (fast) {
			// Do a selection per dimension and do a boolean conjunction on the
			// selection
			BitSet selected = new BitSet(n);
			selected.set(0, n);

			BitSet dimSelected;
			Selection selection = new Selection(n, selectionAlpha);
			for (int i = 0; i < shuffledDimensions.length - 1; i++) {
				dimSelected = selection.selectRandomBlock(data[shuffledDimensions[i]]);
				// boolean conjunction
				selected.and(dimSelected);
			}

			int l = selected.cardinality();
			dimData = new double[l];
			weights = new double[l];
			int j = 0;
			double[] lastData = data[shuffledDimensions[shuffledDimensions.length - 1]].getData();
			double[] lastWeights = data[shuffledDimensions[shuffledDimensions.length - 1]].getWeights();
			for (int i = 0; i < n; i++) {
				if (selected.get(i)) {
					dimData[j] = lastData[i];
					weights[j] = lastWeights[i];
					j++;
				}
			}
		} else {
			Selection selectedIndexes = new Selection(n, selectionAlpha);
			// Fill the list with all the indexes
			selectedIndexes.fillRange();

			for (int i = 0; i < shuffledDimensions.length - 1; i++) {
				// Get all the data for the specific dimension that is selected
				dimData = getSelectedData(shuffledDimensions[i], selectedIndexes);
				weights = getSelectedWeights(selectedIndexes);
				// Reduce the number of indexes according to a new selection in
				// the current dimension
				selectedIndexes.selectWithWeights(dimData, weights);
			}

			// Get the selected data from the last dimension and apply weights
			dimData = getSelectedData(shuffledDimensions[shuffledDimensions.length - 1], selectedIndexes);
			weights = getSelectedWeights(selectedIndexes);

		}
		return new DataBundle(dimData, weights);
	}

	/**
	 * This method is mainly for the visualisation. It contains duplicated code
	 * from above.
	 * 
	 * @param shuffledDimensions
	 * @param selectionAlpha
	 * @return
	 */
	public Selection getSliceIndexes(int[] shuffledDimensions, double selectionAlpha) {
		if (data == null) {
			getAndSortData();
		}

		int n = getNumberOfElements();
		if (n == 0) {
			return null;
		}

		if (fast) {
			// Do a selection per dimension and do a boolean conjunction on the
			// selection
			BitSet selected = new BitSet(n);
			selected.set(0, n);

			BitSet dimSelected;
			Selection selection = new Selection(n, selectionAlpha);
			for (int i = 0; i < shuffledDimensions.length - 1; i++) {
				dimSelected = selection.selectRandomBlock(data[i]);
				// boolean conjunction
				selected.and(dimSelected);
			}

			int l = selected.cardinality();
			Selection selectedIndexes = new Selection(l, selectionAlpha);
			int j = 0;
			for (int i = 0; i < n; i++) {
				if (selected.get(i)) {
					selectedIndexes.getIndexes()[j] = i;
					j++;
				}
			}
			return selectedIndexes;
		} else {
			Selection selectedIndexes = new Selection(n, selectionAlpha);
			// Fill the list with all the indexes
			selectedIndexes.fillRange();

			double[] dimData;
			double[] weights;

			for (int i = 0; i < shuffledDimensions.length - 1; i++) {
				// Get all the data for the specific dimension that is selected
				dimData = getSelectedData(shuffledDimensions[i], selectedIndexes);
				weights = getSelectedWeights(selectedIndexes);
				// Reduce the number of indexes according to a new selection in
				// the current dimension
				selectedIndexes.selectWithWeights(dimData, weights);
			}
			return selectedIndexes;
		}
	}

	private void getAndSortData() {
		data = getData();
		if (fast) {
			for (int i = 0; i < data.length; i++) {
				data[i].sort();
			}
		}
	}

	/**
	 * Returns the selected data, indicated by the indexes. The Databundles
	 * should not be sorted!
	 * 
	 * @param dimension
	 *            The dimension the data should be taken from
	 * @param selectedIndexes
	 *            The {@link Selection}
	 * @return The data from the given dimension with the given indexes in form
	 *         of a {@link DataBundle}.
	 */
	public double[] getSelectedData(int dimension, Selection selectedIndexes) {
		double[] origData = data[dimension].getData();
		int l = selectedIndexes.size();
		double[] selectedData = new double[l];
		for (int i = 0; i < l; i++) {
			selectedData[i] = origData[selectedIndexes.getIndex(i)];
		}

		return selectedData;
	}

	/**
	 * Returns the selected weights, indicated by the indexes. The Databundles
	 * should not be sorted!
	 * 
	 * @param selectedIndexes
	 *            The {@link Selection}
	 * @return The weights with the given indexes in form
	 *         of a {@link DataBundle}.
	 */
	public double[] getSelectedWeights(Selection selectedIndexes) {
		double[] origWeights = data[0].getWeights();
		int l = selectedIndexes.size();
		double[] weights = new double[l];
		for (int i = 0; i < l; i++) {
			weights[i] = origWeights[selectedIndexes.getIndex(i)];
		}

		return weights;
	}
}
