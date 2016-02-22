package fullsystem;

import java.util.ArrayList;

import changechecker.ChangeChecker;
import environment.Stopwatch;
import pruning.AbstractPruner;
import pruning.TopDownPruner;
import streamdatastructures.CorrelationSummary;
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
	 * The @link{Contrast} instance.
	 */
	private Contrast contrastEvaluator;

	/**
	 * The {@link SubspaceBuilder} instance.
	 */
	private SubspaceBuilder subspaceBuilder;

	/**
	 * The {@link ChangeChecker} instance.
	 */
	private ChangeChecker changeChecker;

	/**
	 * The @link{Callback}s to notify on changes.
	 */
	private ArrayList<Callback> callbacks;

	/**
	 * The {@link AbstractPruner} instance.
	 */
	private AbstractPruner pruner;

	/**
	 * The {@link CorrelationSummary} to calculate the Pearsons's correlation
	 * coefficient for pairs of dimensions.
	 */
	private CorrelationSummary correlationSummary;

	/**
	 * The {@link Stopwatch} instance.
	 */
	private Stopwatch stopwatch;

	/**
	 * Creates an instance of this class.
	 * 
	 * 
	 * @param epsilon
	 *            The deviation that is allowed for contrast values between two
	 *            evaluation without starting a full new build of the correlated
	 *            subspaces.
	 * @param threshold
	 *            The threshold for the contrast. {@link Subspace}s with a
	 *            contrast above or equal to the threshold may be considered as
	 *            correlated.
	 * @param pruningDifference
	 *            The difference in contrast allowed to prune a {@link Subspace}
	 *            , if a correlated super-space exists.
	 * @param contrastEvaluator
	 *            The {@link Contrast} instance
	 * @param subspaceBuilder
	 *            The {@link SubspaceBuilder} instance
	 * @param changeChecker
	 *            The {@link ChangeChecker} instance
	 * @param callback
	 *            The {@link Callback} instance
	 * @param correlationSummary
	 *            The {@link CorrelationSummary}
	 * @param stopwatch
	 *            The {@link Stopwatch} instance
	 */
	public StreamHiCS(double epsilon, double threshold, double pruningDifference, Contrast contrastEvaluator,
			SubspaceBuilder subspaceBuilder, ChangeChecker changeChecker, Callback callback,
			CorrelationSummary correlationSummary, Stopwatch stopwatch) {
		correlatedSubspaces = new SubspaceSet();
		if (epsilon < 0) {
			throw new IllegalArgumentException("Non-positive input value.");
		}
		this.epsilon = epsilon;
		this.threshold = threshold;
		this.contrastEvaluator = contrastEvaluator;
		this.subspaceBuilder = subspaceBuilder;
		this.changeChecker = changeChecker;
		this.callbacks = new ArrayList<Callback>();
		if (pruningDifference >= 0) {
			this.pruner = new TopDownPruner(pruningDifference);
		}
		this.correlationSummary = correlationSummary;
		this.stopwatch = stopwatch;
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
	 * Returns the internal {@link Stopwatch}.
	 * 
	 * @return The {@link Stopwatch}
	 */
	public Stopwatch getStopwatch() {
		return this.stopwatch;
	}

	/**
	 * Sets the {@link Callback}.
	 * 
	 * @param callback
	 *            The {@link Callback}.
	 */
	public void addCallback(Callback callback) {
		this.callbacks.add(callback);
	}

	/**
	 * Add a new {@link Instance}.
	 * 
	 * @param instance
	 *            The {@link Instance} to be added.
	 */
	public void add(Instance instance) {
		stopwatch.start("Adding");
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
		if (correlationSummary != null) {
			correlationSummary.addInstance(instance);
		}
		stopwatch.stop("Adding");
		changeChecker.poll();
	}

	/**
	 * Clears all stored information learnt from the stream.
	 */
	public void clear() {
		correlatedSubspaces.clear();
		contrastEvaluator.clear();
		if (correlationSummary != null) {
			correlationSummary.clear();
		}
	}

	/**
	 * Carries out an evaluation of the stored correlated {@link Subspace}s and
	 * searches for new ones.
	 * 
	 * @return True, if the correlated subspaces were updated, false otherwise.
	 */
	public boolean evaluateCorrelatedSubspaces() {
		boolean update = false;
		if (correlatedSubspaces.isEmpty()) {
			// Find new correlated subspaces
			correlatedSubspaces = subspaceBuilder.buildCorrelatedSubspaces();
			if (pruner != null) {
				correlatedSubspaces = pruner.prune(correlatedSubspaces);
			}
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
				// are subspace to another subspace with higher contrast are
				// discarded.
				if (pruner != null) {
					correlatedSubspaces = pruner.prune(correlatedSubspaces);
				}
			}
		}

		return update;
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
		stopwatch.start("Evaluation");
		boolean updated = evaluateCorrelatedSubspaces();
		stopwatch.stop("Evaluation");
		if (updated) {
			// Notify the callbacks
			for (Callback callback : callbacks) {
				callback.onAlarm();
			}
		}
	}

	/**
	 * Checks whether the given {@link Subspace} is still valid, i.e. its
	 * contrast is calculated and checked if it is above or equal to the
	 * threshold and it has not changed more than epsilon.
	 * 
	 * @param subspace
	 *            The {@link Subspace}
	 * @return True, if it is still valid, false otherwise.
	 */
	public boolean isValidSubspace(Subspace subspace) {
		double contrast = contrastEvaluator.evaluateSubspaceContrast(subspace);
		if (Math.abs(Math.abs(contrast - subspace.getContrast())) > epsilon || contrast < threshold) {
			return false;
		}
		return true;
	}
}