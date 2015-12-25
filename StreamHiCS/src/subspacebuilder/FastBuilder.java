package subspacebuilder;

import org.apache.commons.math3.util.MathArrays;

import fullsystem.Contrast;
import subspace.Subspace;
import subspace.SubspaceSet;

/**
 * This class represents a {@link SubspaceBuilder} with a polynomial worst-case.
 * 
 * @author Vincent
 *
 */
public class FastBuilder extends SubspaceBuilder {

	/**
	 * A {@link SubspaceSet} containing the candidates for correlated
	 * {@link Subspace}s.
	 */
	private SubspaceSet correlatedSubspaces;

	/**
	 * The number of dimensions of the full space.
	 */
	private int numberOfDimensions;

	/**
	 * The minimum contrast value a {@link Subspace} must have to be a candidate
	 * for the correlated subspaces. Note that, even if a subspace's contrast
	 * exceeds the threshold it might not be chosen due to the cutoff.
	 */
	private double threshold;

	/**
	 * The number of subspace candidates should be kept after each apriori step.
	 * The cutoff value must be positive. The threshold must be positive.
	 */
	private int cutoff;

	/**
	 * The @link{Contrast} instance.
	 */
	private Contrast contrastEvaluator;

	/**
	 * Creates an instance of this class.
	 * 
	 * @param numberOfDimensions
	 *            THe number of dimensions of the full space
	 * @param threshold
	 *            The threshold
	 * @param cutoff
	 *            The cutoff
	 * @param contrastEvaluator
	 *            The {@link Contrast} instance
	 */
	public FastBuilder(int numberOfDimensions, double threshold, int cutoff, Contrast contrastEvaluator) {
		this.correlatedSubspaces = new SubspaceSet();
		this.numberOfDimensions = numberOfDimensions;
		this.threshold = threshold;
		this.cutoff = cutoff;
		this.contrastEvaluator = contrastEvaluator;
	}

	@Override
	public SubspaceSet buildCorrelatedSubspaces() {
		correlatedSubspaces.clear();
		double contrast = 0;
		// Create all 2-dimensional candidates
		for (int i = 0; i < numberOfDimensions - 1; i++) {
			for (int j = i + 1; j < numberOfDimensions; j++) {
				Subspace s = new Subspace();
				s.addDimension(i);
				s.addDimension(j);
				// Only use subspace for the further process which are
				// correlated
				contrast = contrastEvaluator.evaluateSubspaceContrast(s);
				s.setContrast(contrast);
				if (contrast >= threshold) {
					correlatedSubspaces.addSubspace(s);
				}
			}
		}

		if (!correlatedSubspaces.isEmpty()) {
			// Count the occurence of the dimensions
			double[] occurenceCounts = new double[numberOfDimensions];
			double[] dimensions = new double[numberOfDimensions];
			for (int i = 0; i < numberOfDimensions; i++) {
				dimensions[i] = i;
			}

			for (Subspace s : correlatedSubspaces.getSubspaces()) {
				for (int i = 0; i < s.size(); i++) {
					occurenceCounts[s.getDimension(i)]++;
				}
			}

			MathArrays.sortInPlace(occurenceCounts, MathArrays.OrderDirection.DECREASING, dimensions);

			SubspaceSet set = new SubspaceSet();
			Subspace s;
			int k = 0;
			double lastContrast = 0;
			for (Subspace cs : correlatedSubspaces.getSubspaces()) {
				lastContrast = cs.getContrast();
				// Start from the top adding dimensions and check if their
				// contrast grows
				s = cs.copy();
				k = 2;
				for (int i = 0; i < dimensions.length && occurenceCounts[i] > 0 && i < cutoff; i++) {
					s.addDimension((int) dimensions[i]);
					// Check if a new dimension was added
					if (s.size() == k + 1) {
						k++;
						contrast = contrastEvaluator.evaluateSubspaceContrast(s);
						if (contrast > lastContrast) {
							lastContrast = contrast;
							s.setContrast(contrast);
						} else {
							s.discardDimension(s.size() - 1);
							k--;
						}
					}
				}
				s.sort();
				set.addSubspace(s);
			}

			// Add all the new found subspaces to the correlated subspaces
			correlatedSubspaces.addSubspaces(set);
		}
		return correlatedSubspaces;
	}
}
