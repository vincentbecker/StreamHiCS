package subspacebuilder;

import fullsystem.Contrast;
import streamdatastructures.CorrelationSummary;
import subspace.Subspace;
import subspace.SubspaceSet;

/**
 * This class represents a {@link SubspaceBuilder} using an Apriori-like
 * procedure.
 * 
 * @author Vincent
 *
 */
public class AprioriBuilder extends SubspaceBuilder {

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
	 * exceeds the threshold it might not be chosen due to the cutoff. The
	 * threshold must be positive.
	 */
	private double threshold;

	/**
	 * The number of subspace candidates should be kept after each apriori step.
	 * The cutoff value must be positive.
	 */
	private int cutoff;

	/**
	 * The {@link Contrast} evaluator.
	 */
	private Contrast contrastEvaluator;

	/**
	 * The {@link CorrelationSummary} to calculate the Pearsons's correlation
	 * coefficient for pairs of dimensions.
	 */
	private CorrelationSummary correlationSummary;
	//private Stopwatch stopwatch;

	/**
	 * Creates an instance of this class.
	 * 
	 * @param numberOfDimensions
	 *            The number of dimensions of the full space
	 * @param threshold
	 *            The threshold
	 * @param cutoff
	 *            The cutoff
	 * @param pruningDifference
	 *            The pruning difference
	 * @param contrastEvaluator
	 *            The {@link Contrast} instance
	 * @param correlationSummary
	 *            The {@link CorrelationSummary}
	 */
	public AprioriBuilder(int numberOfDimensions, double threshold, int cutoff,
			Contrast contrastEvaluator, CorrelationSummary correlationSummary) {
		this.correlatedSubspaces = new SubspaceSet();
		this.numberOfDimensions = numberOfDimensions;
		this.threshold = threshold;
		this.cutoff = cutoff;
		this.contrastEvaluator = contrastEvaluator;
		this.correlationSummary = correlationSummary;
	}

	@Override
	public SubspaceSet buildCorrelatedSubspaces() {
		correlatedSubspaces.clear();
		SubspaceSet c_K = new SubspaceSet();
		//double contrast = 0;
		// Create all 2-dimensional candidates
		//stopwatch.start("2D-contrast");
		/*
		 * for (int i = 0; i < numberOfDimensions - 1; i++) { for (int j = i +
		 * 1; j < numberOfDimensions; j++) { Subspace s = new Subspace();
		 * s.addDimension(i); s.addDimension(j); // Only use subspaces for the
		 * further process which are // correlated contrast =
		 * contrastEvaluator.evaluateSubspaceContrast(s);
		 * s.setContrast(contrast); if (contrast >= threshold) {
		 * c_K.addSubspace(s); } } }
		 */
		double[][] coefficientMatrix = correlationSummary.getCorrelationMatrix();
		for (int i = 0; i < numberOfDimensions - 1; i++) {
			for (int j = i + 1; j < numberOfDimensions; j++) {
				if (coefficientMatrix[i][j] >= threshold) {
					Subspace s = new Subspace();
					s.addDimension(i);
					s.addDimension(j);
					s.setContrast(contrastEvaluator.evaluateSubspaceContrast(s));
					c_K.addSubspace(s);
				}
			}
		}
		//stopwatch.stop("2D-contrast");

		// Select cutoff subspaces
		c_K.selectTopK(cutoff);

		// Add the left over 2D subspaces to the correlated subspaces
		correlatedSubspaces.addSubspaces(c_K);

		// Carry out apriori algorithm
		aprioriFull(c_K);
		// aprioriParallel(c_K);

		return correlatedSubspaces;
	}

	/**
	 * Carries out the apriori-algorithm recursively.
	 * 
	 * @param c_K
	 *            The current candidate set for correlated subspaces in the
	 *            recursion.
	 */
	private void apriori(SubspaceSet c_K) {
		SubspaceSet c_Kplus1 = new SubspaceSet();
		c_K.sort();
		// The subspaces in the set are sorted. To speed up the process we can
		// stop looking
		// for a merge as soon as we have found the first "no merge" case
		boolean continueMerging = true;
		double contrast = 0;
		for (int i = 0; i < c_K.size() - 1; i++) {
			continueMerging = true;
			for (int j = i + 1; j < c_K.size() && continueMerging; j++) {
				// Creating new candidates
				Subspace kPlus1Candidate = Subspace.merge(c_K.getSubspace(i), c_K.getSubspace(j));

				if (kPlus1Candidate != null) {
					// Calculate the contrast of the subspace
					contrast = contrastEvaluator.evaluateSubspaceContrast(kPlus1Candidate);
					kPlus1Candidate.setContrast(contrast);
					if (contrast >= threshold) {
						c_Kplus1.addSubspace(kPlus1Candidate);
					}
				} else {
					continueMerging = false;
				}
			}
		}
		if (!c_Kplus1.isEmpty()) {
			// Select the subspaces with highest contrast
			c_Kplus1.selectTopK(cutoff);
			correlatedSubspaces.addSubspaces(c_Kplus1);
			// Recurse
			apriori(c_Kplus1);
		}
	}

	/**
	 * Does not only check from the beginning of a set for overlap of two
	 * {@link Subspace}s in an iteration.
	 * 
	 * @param c_K
	 *            The {@link SubspaceSet} containing the current candidates for
	 *            correlated subspaces.
	 */
	private void aprioriFull(SubspaceSet c_K) {
		SubspaceSet c_Kplus1 = new SubspaceSet();
		c_K.sort();
		double contrast = 0;
		// double meanBaseContrasts = 0;
		for (int i = 0; i < c_K.size() - 1; i++) {
			for (int j = i + 1; j < c_K.size(); j++) {
				// Creating new candidates
				// meanBaseContrasts = (c_K.getSubspace(i).getContrast() +
				// c_K.getSubspace(j).getContrast()) / 2;
				Subspace kPlus1Candidate = Subspace.mergeFull(c_K.getSubspace(i), c_K.getSubspace(j));
				if (kPlus1Candidate != null && !c_Kplus1.contains(kPlus1Candidate)) {
					// Calculate the contrast of the subspace
					contrast = contrastEvaluator.evaluateSubspaceContrast(kPlus1Candidate);
					kPlus1Candidate.setContrast(contrast);
					// contrast > meanBaseContrasts - 0.5*pruningDifference &&
					if (contrast >= threshold) {
						c_Kplus1.addSubspace(kPlus1Candidate);
					}
				}
			}
		}
		if (!c_Kplus1.isEmpty()) {
			// Select the subspaces with highest contrast
			c_Kplus1.selectTopK(cutoff);
			correlatedSubspaces.addSubspaces(c_Kplus1);
			// Recurse
			aprioriFull(c_Kplus1);
		}
	}

	/**
	 * Parallel version.
	 * 
	 * @param c_K
	 */
	private void aprioriParallel(SubspaceSet c_K) {
		SubspaceSet c_Kplus1 = new SubspaceSet();
		SubspaceSet temp = new SubspaceSet();
		c_K.sort();
		// The subspaces in the set are sorted. To speed up the process we can
		// stop looking
		// for a merge as soon as we have found the first "no merge" case
		boolean continueMerging = true;
		for (int i = 0; i < c_K.size() - 1; i++) {
			continueMerging = true;
			for (int j = i + 1; j < c_K.size() && continueMerging; j++) {
				// Creating new candidates
				Subspace kPlus1Candidate = Subspace.mergeFull(c_K.getSubspace(i), c_K.getSubspace(j));
				if (kPlus1Candidate != null) {
					temp.addSubspace(kPlus1Candidate);
				} else {
					continueMerging = false;
				}
			}
		}

		// Parallel execution of the contrast calculation
		temp.getSubspaces().parallelStream().forEach(candidate -> {
			candidate.setContrast(contrastEvaluator.evaluateSubspaceContrast(candidate));

		});

		// Add the subspaces greater or equal to the threshold to c_Kplus1
		for (Subspace candidate : temp.getSubspaces()) {
			if (candidate.getContrast() >= threshold) {
				c_Kplus1.addSubspace(candidate);
			}
		}

		if (!c_Kplus1.isEmpty()) {
			// Select the subspaces with highest contrast
			c_Kplus1.selectTopK(cutoff);
			correlatedSubspaces.addSubspaces(c_Kplus1);
			// Recurse
			aprioriParallel(c_Kplus1);
		}
	}
}
