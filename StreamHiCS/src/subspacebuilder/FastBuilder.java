package subspacebuilder;

import java.util.ArrayList;

import org.apache.commons.math3.util.MathArrays;

import contrast.Contrast;
import subspace.Subspace;
import subspace.SubspaceSet;

public class FastBuilder extends SubspaceBuilder {

	SubspaceSet correlatedSubspaces;
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
	 * The difference in contrast allowed to prune a {@link Subspace}.
	 */
	private double pruningDifference;
	/**
	 * The @link{Contrast} evaluator.
	 */
	private Contrast contrastEvaluator;

	public FastBuilder(int numberOfDimensions, double threshold, double pruningDifference, Contrast contrastEvaluator) {
		this.correlatedSubspaces = new SubspaceSet();
		this.numberOfDimensions = numberOfDimensions;
		this.threshold = threshold;
		this.pruningDifference = pruningDifference;
		this.contrastEvaluator = contrastEvaluator;
	}

	@Override
	public SubspaceSet buildCorrelatedSubspaces() {
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
				for (int i = 0; i < dimensions.length && occurenceCounts[i] > 0; i++) {
					s.addDimension((int) dimensions[i]);
					// Check if a new dimension was added
					if (s.size() == k + 1) {
						k++;
						contrast = contrastEvaluator.evaluateSubspaceContrast(s);
						if (contrast > lastContrast) {
							lastContrast = contrast;
						}else{
							s.discardDimension(s.size() - 1);
							k--;
						}
					}
				}
				s.setContrast(contrast);
				s.sort();
				set.addSubspace(s);
			}

			// Add all the new found subspaces to the correlated subspaces
			correlatedSubspaces.addSubspaces(set);

			// Carry out pruning as the last step. All those subspaces which are
			// subspace to another subspace with higher contrast are discarded.
			prune();
		}
		return correlatedSubspaces;
	}

	/**
	 * If a {@link Subspace} is a subspace of another subspace with a higher
	 * contrast value, then it is discarded.
	 */
	private void prune() {
		// First pruning step raises the threshold and filters the list
		Subspace s;
		for (int i = 0; i < correlatedSubspaces.size(); i++) {
			s = correlatedSubspaces.getSubspace(i);
			if (s.getContrast() < threshold + pruningDifference) {
				correlatedSubspaces.removeSubspace(s);
			}
		}
		
		//Second pruning step checks fo rcontained subspaces
		ArrayList<Integer> discard = new ArrayList<Integer>();
		int l = correlatedSubspaces.size();
		Subspace si;
		Subspace sj;
		boolean discarded;
		for (int i = 0; i < l; i++) {
			discarded = false;
			si = correlatedSubspaces.getSubspace(i);
			for (int j = 0; j < l && !discarded; j++) {
				if (i != j) {
					sj = correlatedSubspaces.getSubspace(j);
					// If the correlated subspace contains a superset that has
					// at least (nearly) the same contrast we discard the
					// current subspace
					if (si.isSubspaceOf(sj) && si.getContrast() <= (sj.getContrast() + pruningDifference)) {
						discard.add(i);
						discarded = true;
					}
				}
			}
		}
		SubspaceSet prunedSet = new SubspaceSet();
		for (int i = 0; i < l; i++) {
			if (!discard.contains(i)) {
				prunedSet.addSubspace(correlatedSubspaces.getSubspace(i));
			}
		}
		correlatedSubspaces = prunedSet;
	}
}
