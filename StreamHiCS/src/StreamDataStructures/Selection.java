package streamdatastructures;

import java.util.BitSet;
import java.util.Random;
import org.apache.commons.math3.util.MathArrays;

/**
 * This class represents a selection of indexes.
 * 
 * @author Vincent
 *
 */
public class Selection {

	/**
	 * The indexes currently held.
	 */
	private double[] indexes;
	/**
	 * The selection alpha.
	 */
	private double selectionAlpha;
	/**
	 * A generator for random numbers.
	 */
	private Random generator;

	/**
	 * Creates a {@link Selection} object.
	 * 
	 * @param initialSize
	 *            The initial size of the index array.
	 * @param selectionAlpha
	 *            The selection alpha.
	 */
	public Selection(int initialSize, double selectionAlpha) {
		indexes = new double[initialSize];
		this.selectionAlpha = selectionAlpha;
		generator = new Random();
	}

	/**
	 * Returns the number of indexes currently held.
	 * 
	 * @return The number of indexes.
	 */
	public int size() {
		return indexes.length;
	}

	/**
	 * Returns the index at the given position.
	 * 
	 * @param i
	 *            The position
	 * @return The index at the given position.
	 */
	public int getIndex(int i) {
		return (int) indexes[i];
	}

	/**
	 * Returns the index array.
	 * 
	 * @return The index array.
	 */
	public double[] getIndexes() {
		return indexes;
	}

	/**
	 * Checks whether this {@link Selection} contains the given index.
	 * 
	 * @param i
	 *            The index to searh for
	 * @return True, if this {@link Selection} contains the index, false
	 *         otherwise.
	 */
	public boolean contains(int i) {
		for (Double d : indexes) {
			if (d == i) {
				return true;
			}
		}
		return false;
	}

	/**
	 * Fills the index array with the range beginning at 0.
	 */
	public void fillRange() {
		for (int i = 0; i < indexes.length; i++) {
			indexes[i] = i;
		}
	}

	/**
	 * Selects a range of indexes according to given data. The indexes are
	 * arranged so that they correspond to the sorted data. Then a random block
	 * selection of the size selectionAlpha*(size of data) is drawn,
	 * corresponding to a block of values in the sorted data. The indexes of
	 * that selection are stored.
	 * 
	 * @param data
	 *            The data.
	 */
	public void select(double[] data) {
		int l = data.length;
		int selectionSize = (int) (l * selectionAlpha);

		if (selectionSize < l) {
			// Sort the data and arrange the indexes correspondingly
			MathArrays.sortInPlace(data, indexes);
			// Start at a random point and take the selectionSize
			int startingPoint = generator.nextInt(l - selectionSize + 1);
			int endPoint = startingPoint + selectionSize - 1;
			if (startingPoint < 0 || endPoint > indexes.length - 1) {
				throw new IllegalArgumentException("Selection outside of range: [" + startingPoint + endPoint + "]");
			}
			if (data[startingPoint] == data[endPoint]) {
				// The special case that all the data values selected are the
				// same. This case needs special handling.
				// System.out.println("Selection.selectRange(): Special
				// handling!");
				// Broadening the range if possible, until data values on the
				// outside of the range differ
				while (data[startingPoint] == data[endPoint] && (endPoint - startingPoint) < indexes.length - 1) {
					if (startingPoint > 0) {
						startingPoint--;
					}
					if (endPoint < indexes.length - 1) {
						endPoint++;
					}
				}
			}
			double[] newIndexes = new double[endPoint - startingPoint + 1];
			for (int i = startingPoint; i <= endPoint; i++) {
				newIndexes[i - startingPoint] = indexes[i];
			}

			// Set the indexes to the new range of indexes
			indexes = newIndexes;
		}

	}

	/**
	 * Selects a random range where the total weight of the range is a fraction
	 * (selectionAlpha) of the total weight.
	 * 
	 * @param data
	 *            The data
	 * @param weights
	 *            The weights
	 */
	public void selectWithWeights(double[] data, double[] weights) {
		double totalWeight = 0;
		for (int i = 0; i < data.length; i++) {
			totalWeight += weights[i];
		}
		double selectionSize = totalWeight * selectionAlpha;

		// Sort the data and arrange the indexes correspondingly
		MathArrays.sortInPlace(data, indexes, weights);

		// Start at a random point and take the selectionSize
		int startingPoint = generator.nextInt(data.length);

		// Select a block around the starting point
		int lower = startingPoint;
		int upper = startingPoint;
		double lowerWeight = 0;
		double upperWeight = 0;
		double accumulatedWeight = weights[startingPoint];
		double selectionPerSide = (selectionSize - weights[startingPoint]) / 2;
		boolean searchOn = false;
		while (accumulatedWeight < selectionSize - 0.000001) {
			while ((lowerWeight < selectionPerSide || (searchOn && accumulatedWeight < selectionSize)) && lower > 0) {
				lower--;
				lowerWeight += weights[lower];
				accumulatedWeight += weights[lower];
			}
			while ((upperWeight < selectionPerSide || (searchOn && accumulatedWeight < selectionSize))
					&& upper < weights.length - 1) {
				upper++;
				upperWeight += weights[upper];
				accumulatedWeight += weights[upper];
			}
			searchOn = true;
		}

		// Since the Kolmogorov-Smirnov-Test needs at least two samples we take
		// at least one other another if there is one a single one selected
		if (upper - lower == 0) {
			// System.out.println("Only one sample.");
			if (lower > 0) {
				lower--;
			}
			if (upper < data.length - 1) {
				upper++;
			}
		}

		// Broadening the range until the values on the edges differ from the
		// next value outside. This is done to from a correct slice.
		while (lower > 0 && data[lower - 1] == data[lower]) {
			lower--;
			// System.out.println("Selection.selectRangeWithWeights(): Special
			// handling!");
		}
		while (upper < data.length - 1 && data[upper] == data[upper + 1]) {
			upper++;
			// System.out.println("Selection.selectRangeWithWeights(): Special
			// handling!");
		}

		// System.out.println("Lower: " + lower + " Upper: " + upper);
		double[] newIndexes = new double[upper - lower + 1];
		for (int i = lower; i <= upper; i++) {
			newIndexes[i - lower] = indexes[i];
		}
		indexes = newIndexes;
	}

	/**
	 * Selects a random block of indexes where the data is sorted. The weight of
	 * the selected points is a fraction (selectionAlpha) of the total weight.
	 * 
	 * @param databundle The {@link DataBundle} containing the data and the weights. 
	 * 
	 * @return A {@link BitSet} (i.e. boolean vector) where the entries are set true at the selected positions and false otherwise. 
	 */
	public BitSet selectRandomBlock(DataBundle databundle) {
		double[] indexes = databundle.getSortedIndexes();
		double[] data = databundle.getSortedData();
		double[] weights = databundle.getSortedWeights();

		int n = data.length;
		double totalWeight = 0;
		for (int i = 0; i < n; i++) {
			totalWeight += weights[i];
		}
		double selectionSize = totalWeight * selectionAlpha;

		// Start at a random point and take the selectionSize
		int startingPoint = generator.nextInt(n);

		// Select a block around the starting point
		int lower = startingPoint;
		int upper = startingPoint;
		double lowerWeight = 0;
		double upperWeight = 0;
		double accumulatedWeight = weights[startingPoint];
		double selectionPerSide = (selectionSize - weights[startingPoint]) / 2;
		boolean searchOn = false;
		while (accumulatedWeight < selectionSize) {
			while ((lowerWeight < selectionPerSide || (searchOn && accumulatedWeight < selectionSize)) && lower > 0) {
				lower--;
				lowerWeight += weights[lower];
				accumulatedWeight += weights[lower];
			}
			while ((upperWeight < selectionPerSide || (searchOn && accumulatedWeight < selectionSize))
					&& upper < n - 1) {
				upper++;
				upperWeight += weights[upper];
				accumulatedWeight += weights[upper];
			}
			searchOn = true;
		}

		// Since the Kolmogorov-Smirnov-Test needs at least two samples we take
		// at least one other another if there is one a single one selected
		if (upper - lower == 0) {
			// System.out.println("Only one sample.");
			if (lower > 0) {
				lower--;
			}
			if (upper < n - 1) {
				upper++;
			}
		}

		// Broadening the range until the values on the edges differ from the
		// next value outside. This is done to from a correct slice.
		while (lower > 0 && data[lower - 1] == data[lower]) {
			lower--;
		}
		while (upper < n - 1 && data[upper] == data[upper + 1]) {
			upper++;
		}
		// All boolean entries initialized to false
		BitSet selected = new BitSet(n);
		for (int i = lower; i <= upper; i++) {
			selected.set((int) indexes[i]);
		}

		return selected;

	}

	/**
	 * Returns a String representation of this object.
	 * 
	 * @return A String representation of this object.
	 */
	public String toString() {
		String rep = "";
		for (int i = 0; i < indexes.length - 1; i++) {
			rep += ((int) indexes[i]) + ", ";
		}
		rep += (int) indexes[indexes.length - 1];
		return rep;
	}
}
