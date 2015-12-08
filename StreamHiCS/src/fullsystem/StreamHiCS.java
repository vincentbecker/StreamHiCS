package fullsystem;

import changechecker.ChangeChecker;
import contrast.Contrast;
import pruning.AbstractPruner;
import pruning.TopDownPruner;
import subspace.Subspace;
import subspace.SubspaceSet;
import subspacebuilder.SubspaceBuilder;
import weka.core.DenseInstance;
import weka.core.Instance;

public class StreamHiCS implements Callback {
	/**
	 * The set of the currently correlated {@link Subspace}s.
	 */
	private SubspaceSet correlatedSubspaces;
	/**
	 * Determines how much the contrast values of {@link Subspace]s are allowed
	 * to deviate from the last evaluation. If the deviation exceeds epsilon the
	 * correlated {Subspace}s are newly built. Epsilon must be positive.
	 */
	private double epsilon;
	/**
	 * The minimum contrast value a {@link Subspace} must have to be a candidate
	 * for the correlated subspaces.
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
	private SubspaceBuilder subspaceBuilder;
	private ChangeChecker changeChecker;
	/**
	 * The @link{Callback} to notify on changes.
	 */
	private Callback callback;
	private AbstractPruner pruner;

	/**
	 * Creates a {@link StreamHiCS} object with the specified update interval.
	 * 
	 * @param numberOfDimensions
	 *            The number of dimensions of the full space.
	 * @param epsilon
	 *            The deviation that is allowed for contrast values between two
	 *            evaluation without starting a full new build of the correlated
	 *            subspaces.
	 * @param threshold
	 *            The threshold for the contrast. {@link Subspace}s with a
	 *            contrast above or equal to the threshold may be considered as
	 *            correlated.
	 */
	public StreamHiCS(double epsilon, double threshold, double pruningDifference, Contrast contrastEvaluator,
			SubspaceBuilder subspaceBuilder, ChangeChecker changeChecker, Callback callback) {
		correlatedSubspaces = new SubspaceSet();
		if (epsilon < 0) {
			throw new IllegalArgumentException("Non-positive input value.");
		}
		this.epsilon = epsilon;
		this.threshold = threshold;
		this.pruningDifference = pruningDifference;
		this.contrastEvaluator = contrastEvaluator;
		this.subspaceBuilder = subspaceBuilder;
		this.changeChecker = changeChecker;
		this.callback = callback;
		// this.pruner = new SimplePruner(pruningDifference);
		this.pruner = new TopDownPruner(pruningDifference);
	}

	/**
	 * Returns the correlated {@link Subspace}s.
	 * 
	 * @return The {@link SubspaceSet} containing the currently correlated
	 *         {@link Subspace}s.
	 */
	public SubspaceSet getCurrentlyCorrelatedSubspaces() {
		return correlatedSubspaces;
	}

	/**
	 * Returns the number of elements in the summarisation structure.
	 * 
	 * @return The number of elements in the summarisation structure.
	 */
	public int getNumberOfElements() {
		return contrastEvaluator.getNumberOfElements();
	}

	/**
	 * Sets the callback.
	 * 
	 * @param callback
	 *            The callback.
	 */
	public void setCallback(Callback callback) {
		this.callback = callback;
	}

	/**
	 * Add a new {@link Instance}.
	 * 
	 * @param instance
	 *            The {@link Instance} to be added.
	 */
	public void add(Instance instance) {
		// StreamHiCS is works in an unsupervised fashion. Therefore we create a
		// new instance without a class label, since this will simply be used as
		// an additional dimension.
		// Check if the instance has a class attribute
		if (instance.classIndex() >= 0) {
			int numberOfDimensions = instance.numAttributes() - 1;
			DenseInstance newInst = new DenseInstance(numberOfDimensions);
			for (int i = 0; i < numberOfDimensions; i++) {
				newInst.setValue(i, instance.value(i));
			}
			instance = newInst;
		}
		contrastEvaluator.add(instance);
		changeChecker.poll();
	}

	/**
	 * Clears all stored information learnt from the stream.
	 */
	public void clear() {
		correlatedSubspaces.clear();
		contrastEvaluator.clear();
	}

	/**
	 * Carries out an evaluation of the stored correlated {@link Subspace}s and
	 * searches for new ones.
	 */
	private boolean evaluateCorrelatedSubspaces() {
		boolean update = false;
		if (correlatedSubspaces.isEmpty()) {
			// Find new correlated subspaces
			correlatedSubspaces = subspaceBuilder.buildCorrelatedSubspaces();
			correlatedSubspaces = pruner.prune(correlatedSubspaces);
			if (!correlatedSubspaces.isEmpty()) {
				update = true;
			}
		} else {
			double contrast = 0;
			update = false;
			// This variable is needed to build a workaround of problems with
			// the iterator when removing elements at the same time
			int l = correlatedSubspaces.size();
			SubspaceSet keep = new SubspaceSet();
			for (int i = 0; i < l; i++) {
				Subspace subspace = correlatedSubspaces.getSubspace(i);
				contrast = contrastEvaluator.evaluateSubspaceContrast(subspace);
				// System.out.println(contrast);

				// If contrast has changed more than epsilon or has fallen below
				// the threshold we start a new complete evaluation.
				if (Math.abs(contrast - subspace.getContrast()) > epsilon || contrast < threshold) {
					update = true;
				} else {
					keep.addSubspace(subspace);
				}
				subspace.setContrast(contrast);
			}
			// If a subspace has changed we update the correlated
			// subspaces.
			if (update) {
				correlatedSubspaces.clear();
				correlatedSubspaces.addSubspaces(keep);
				correlatedSubspaces.addSubspaces(subspaceBuilder.buildCorrelatedSubspaces());
				// Carry out pruning as the last step. All those subspaces which
				// are
				// subspace to another subspace with higher contrast are
				// discarded.
				correlatedSubspaces = pruner.prune(correlatedSubspaces);
			}
		}

		return update;

		/*
		 * TODO: Remove // Parallel version List<Double> res =
		 * correlatedSubspaces.parallelStream().map(s -> { return
		 * evaluateSubspaceContrast(s); }).collect(Collectors.toList());
		 * 
		 * for (Double d : res) { if (d < threshold) { update = true; } }
		 */
	}

	/**
	 * Returns a string representation of this object.
	 * 
	 * @return A string representation of this object.
	 */
	public String toString() {
		return correlatedSubspaces.toString();
	}

	@Override
	public void onAlarm() {
		if (evaluateCorrelatedSubspaces()) {
			// Notify the callback
			callback.onAlarm();
		}
	}
}