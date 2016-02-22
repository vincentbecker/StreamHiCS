package changedetection;

import java.util.ArrayList;
import java.util.BitSet;

import fullsystem.Callback;
import fullsystem.StreamHiCS;
import moa.classifiers.AbstractClassifier;
import moa.classifiers.bayes.NaiveBayes;
import moa.classifiers.trees.HoeffdingTree;
import moa.core.Measurement;
import moa.core.ObjectRepository;
import moa.options.FlagOption;
import moa.options.IntOption;
import moa.tasks.TaskMonitor;
import subspace.Subspace;
import subspace.SubspaceSet;
import weka.core.Instance;
import weka.core.Utils;

/**
 * This class represents the top level of the system. It uses StreamHiCS to
 * obtain the correlated {@link Subspace}s and runs DDM concept change detectors
 * on them. Additionally it runs one detector on the full space as well just as
 * a usual change detector would.
 * 
 * @author Vincent
 *
 */
public class SubspaceChangeDetectors extends AbstractClassifier implements Callback, ChangeDetector {

	public enum State {
		IN_CONTROL, WARNING, DRIFT
	};

	/**
	 * The serial version ID.
	 */
	private static final long serialVersionUID = -686978424502737299L;

	/**
	 * The option determining the value of initial point before the correlated
	 * {@link Subspace}s are evaluated the first time.
	 */
	public IntOption initOption = new IntOption("init", 'i',
			"The number of instances after which the correlated subspaces are evaluated the first time.", 500, 0,
			Integer.MAX_VALUE);

	/**
	 * The option determining whether to use the restspace or not.
	 */
	public FlagOption useRestspaceOption = new FlagOption("useRestspace", 'u',
			"Whether the restspace should be included in the change detection process.");

	/**
	 * The option determining whether to forward the arriving {@link Instance}s
	 * to {@link StreamHiCS}. Only for test reasons, should always be set
	 * otherwise.
	 */
	public FlagOption addOption = new FlagOption("add", 'a',
			"Whether arriving instances should be added to streamHiCS.");

	/**
	 * The {@link StreamHiCS} instance.
	 */
	private StreamHiCS streamHiCS;

	/**
	 * The {@link FullSpaceChangeDetector} instance.
	 */
	private FullSpaceChangeDetector fullSpaceChangeDetector;

	/**
	 * The {@link ArrayList} containing the {@link SubspaceChangeDetector}s,
	 * i.e. the change detector running on each correlated {@link Subspace}.
	 */
	private ArrayList<SubspaceChangeDetector> subspaceChangeDetectors;

	/**
	 * A {@link SubspaceChangeDetector} running on all dimensions which are not
	 * contained in the correlated subspaces.
	 */
	private SubspaceChangeDetector restSpaceChangeDetector;

	/**
	 * The number of dimensions.
	 */
	private int numberOfDimensions;

	/**
	 * The number of samples seen.
	 */
	private int numberSamples;

	/**
	 * A flag showing if the an evaluation of the correlated subspace has been
	 * carried out already.
	 */
	private boolean initialized = false;

	/**
	 * The number of instances received before evaluating the correlated
	 * subspaces the first time.
	 */
	private int numberInit;

	/**
	 * The current {@link State}.
	 */
	private State state;

	/**
	 * Whether to use the restspace or not.
	 */
	private boolean useRestspace;

	/**
	 * Whether to forward instances to StreamHiCS or not.
	 */
	private boolean add;

	/**
	 * Creates an instance of this class.
	 * 
	 * @param numberOfDimensions
	 *            The number of dimensions.
	 * @param streamHiCS
	 *            The {@link StreamHiCS} instance.
	 */
	public SubspaceChangeDetectors(int numberOfDimensions, StreamHiCS streamHiCS) {
		this.numberOfDimensions = numberOfDimensions;
		this.streamHiCS = streamHiCS;
	}

	/**
	 * Checks for the warning state. 
	 * @return True, if in warning state, false otherwise. 
	 */
	public boolean isWarningDetected() {
		return (state == State.WARNING);
	}

	/**
	 * Checks for the drift state. 
	 * @return True, if in drift state, false otherwise. 
	 */
	public boolean isChangeDetected() {
		return (state == State.DRIFT);
	}

	@Override
	public void onAlarm() {
		// Change subspaces and reset ChangeDetection
		if (initialized) {
			SubspaceSet correlatedSubspaces = streamHiCS.getCurrentlyCorrelatedSubspaces();
			// System.out.println("SCD: Correlated: " +
			// correlatedSubspaces.toString() + " at " + numberSamples);
			ArrayList<SubspaceChangeDetector> temp = new ArrayList<SubspaceChangeDetector>();
			// Mark which dimensions are contained in subspaces to add detectors
			// for all the single dimensions which are not contained in
			// subspaces.
			BitSet marked = new BitSet(numberOfDimensions);
			boolean found = false;
			for (Subspace s : correlatedSubspaces.getSubspaces()) {
				for (int d : s.getDimensions()) {
					marked.set(d);
				}
				found = false;
				SubspaceChangeDetector scd;
				for (int i = 0; i < subspaceChangeDetectors.size() && !found; i++) {
					scd = subspaceChangeDetectors.get(i);
					// If there is an 'old' change detector running on the
					// subspace already we continue using that
					if (s.equals(scd.getSubspace())) {
						temp.add(scd);
						found = true;
					}
				}
				// If the subspace is new we start a new change detector on it
				if (!found) {
					scd = createSubspaceChangeDetector(s);
					temp.add(scd);
				}
			}
			// Handle left over dimensions
			Subspace restSpace = new Subspace();
			for (int i = 0; i < numberOfDimensions; i++) {
				if (!marked.get(i)) {
					restSpace.addDimension(i);
				}
			}
			if (!restSpace.isEmpty()) {
				if ((useRestspace || correlatedSubspaces.isEmpty()) && (restSpaceChangeDetector == null
						|| !restSpace.equals(restSpaceChangeDetector.getSubspace()))) {
					restSpaceChangeDetector = createRestspaceDetector(restSpace);
				}
			} else {
				restSpaceChangeDetector = null;
			}
			subspaceChangeDetectors = temp;
		}
	}

	@Override
	public void trainOnInstanceImpl(Instance inst) {
		if (add) {
			streamHiCS.add(inst);
		}
		if (numberSamples == numberInit) {
			// Evaluate the correlated subspaces once
			init();
		}
		if (!initialized) {
			fullSpaceChangeDetector.trainOnInstance(inst);
			if (fullSpaceChangeDetector.isWarningDetected()) {
				state = State.WARNING;
			} else if (fullSpaceChangeDetector.isChangeDetected()) {
				state = State.DRIFT;
			} else {
				state = State.IN_CONTROL;
			}

		} else {
			boolean warning = false;
			boolean drift = false;
			ArrayList<Integer> driftDetectedIndexes = new ArrayList<Integer>();
			for (int i = 0; i < subspaceChangeDetectors.size(); i++) {
				SubspaceChangeDetector scd = subspaceChangeDetectors.get(i);
				scd.trainOnInstance(inst);
				if (scd.isWarningDetected()) {
					warning = true;
				} else if (scd.isChangeDetected()) {
					drift = true;
					driftDetectedIndexes.add(i);
					/*
					 * if (streamHiCS.isValidSubspace(scd.getSubspace())) { //
					 * System.out.println( // "cscd: CHANGE in subspace: " + //
					 * scd.getSubspace().toString() + " at " + //
					 * numberSamples); drift = true;
					 * driftDetectedIndexes.add(i); } else { //
					 * System.out.println("Would have been change in //
					 * subspace: " + scd.getSubspace().toString() + " at // " +
					 * numberSamples); streamHiCS.evaluateCorrelatedSubspaces();
					 * }
					 */
				}
			}
			// Change in restspace?
			if (restSpaceChangeDetector != null && (subspaceChangeDetectors.isEmpty() || useRestspace)) {
				restSpaceChangeDetector.trainOnInstance(inst);
				if (restSpaceChangeDetector.isWarningDetected()) {
					warning = true;
				} else if (restSpaceChangeDetector.isChangeDetected()) {
					drift = true;
				}
			}
			if (drift) {
				state = State.DRIFT;
				driftOccurred(driftDetectedIndexes);
			} else if (warning) {
				state = State.WARNING;
			} else {
				state = State.IN_CONTROL;
			}
		}
		numberSamples++;
	}

	/**
	 * The reaction to the detection fo a drift.
	 * 
	 * @param driftDetectedIndexes
	 *            The indexes of the {@link SubspaceChangeDetector}s which
	 *            detected the drift (there may be several).
	 */
	private void driftOccurred(ArrayList<Integer> driftDetectedIndexes) {
		for (int i = 0; i < subspaceChangeDetectors.size(); i++) {
			if (!driftDetectedIndexes.contains(i)) {
				// subspaceChangeDetectors.get(i).changeClassifier();
				subspaceChangeDetectors.get(i).resetLearning();
			}
		}
	}

	/**
	 * Initialise the {@link SubspaceChangeDetector}s.
	 */
	private void init() {
		initialized = true;
		if (subspaceChangeDetectors.isEmpty() && restSpaceChangeDetector == null) {
			onAlarm();
		}
	}

	/**
	 * Creates and initialises a {@link SubspaceChangeDetector} for a given
	 * {@link Subspace}.
	 * 
	 * @param s
	 *            The subspace
	 * @return The {@link SubspaceChangeDetector}.
	 */
	private SubspaceChangeDetector createSubspaceChangeDetector(Subspace s) {
		SubspaceChangeDetector scd = new SubspaceChangeDetector(s);
		AbstractClassifier baseLearner = new HoeffdingTree();
		baseLearner.prepareForUse();
		scd.baseLearnerOption.setCurrentObject(baseLearner);
		scd.prepareForUse();
		return scd;
	}

	/**
	 * The same procedure as above for creating a {@link SubspaceChangeDetector}
	 * for the restspace. The only difference is that {@link NaiveBayes} instead
	 * of the {@link HoeffdingTree} is used.
	 * 
	 * @param restspace The {@link Subspace} representing the restspace
	 * @return A {@link SubspaceChangeDetector} for the restspace. 
	 */
	private SubspaceChangeDetector createRestspaceDetector(Subspace restspace) {
		SubspaceChangeDetector restSpaceChangeDetector = new SubspaceChangeDetector(restspace);
		AbstractClassifier baseLearner = new NaiveBayes();
		baseLearner.prepareForUse();
		restSpaceChangeDetector.baseLearnerOption.setCurrentObject(baseLearner);
		restSpaceChangeDetector.prepareForUse();
		return restSpaceChangeDetector;
	}

	@Override
	public void prepareForUseImpl(TaskMonitor arg0, ObjectRepository arg1) {
		useRestspace = useRestspaceOption.isSet();
		// ChangeDetectors
		this.fullSpaceChangeDetector = new FullSpaceChangeDetector();
		AbstractClassifier baseLearner = new HoeffdingTree();
		// AbstractClassifier baseLearner = new HoeffdingAdaptiveTree();
		// AbstractClassifier baseLearner = new DecisionStump();
		baseLearner.prepareForUse();
		fullSpaceChangeDetector.baseLearnerOption.setCurrentObject(baseLearner);
		fullSpaceChangeDetector.prepareForUse();
		// No correlated subspaces yet.
		this.subspaceChangeDetectors = new ArrayList<SubspaceChangeDetector>();
		this.numberInit = initOption.getValue();
		this.add = addOption.isSet();
		super.prepareForUseImpl(arg0, arg1);
	}

	/**
	 * Returns the number of elements held in {@link StreamHiCS}.
	 * 
	 * @return The number of elements held in {@link StreamHiCS}.
	 */
	public int getNumberOfElements() {
		return streamHiCS.getNumberOfElements();
	}

	/**
	 * Returns the currently correlated {@link Subspace}s in a
	 * {@link SubspaceSet}.
	 * 
	 * @return The currently correlated {@link Subspace}s in a
	 *         {@link SubspaceSet}.
	 */
	public SubspaceSet getCurrentlyCorrelatedSubspaces() {
		return streamHiCS.getCurrentlyCorrelatedSubspaces();
	}

	@Override
	public void resetLearningImpl() {
		fullSpaceChangeDetector.resetLearning();
		numberSamples = 0;
		subspaceChangeDetectors.clear();
		restSpaceChangeDetector = null;
		streamHiCS.clear();
		initialized = false;
	}

	@Override
	public boolean isRandomizable() {
		// TODO Auto-generated method stub
		return false;
	}

	@Override
	public void getModelDescription(StringBuilder arg0, int arg1) {
		// TODO Auto-generated method stub

	}

	@Override
	protected Measurement[] getModelMeasurementsImpl() {
		// TODO Auto-generated method stub
		return null;
	}

	public int getClassPrediction(Instance instance) {
		return Utils.maxIndex(getVotesForInstance(instance));
	}

	@Override
	public double[] getVotesForInstance(Instance instance) {
		if (numberSamples < numberInit) {
			return fullSpaceChangeDetector.getVotesForInstance(instance);
		} else {
			int numberClasses = instance.classAttribute().numValues();
			double[] totalVotes = new double[numberClasses];
			double weight;
			for (SubspaceChangeDetector scd : subspaceChangeDetectors) {
				double[] scdVotes = scd.getVotesForInstance(instance);
				weight = scd.getAccuracy();
				if (weight == 0.0) {
					weight = 0.001;
				}
				for (int i = 0; i < scdVotes.length; i++) {
					totalVotes[i] += scdVotes[i] * weight;
				}
			}

			if (restSpaceChangeDetector != null && (subspaceChangeDetectors.isEmpty() || useRestspace)) {
				double[] restSpaceVotes = restSpaceChangeDetector.getVotesForInstance(instance);
				weight = restSpaceChangeDetector.getAccuracy();
				if (weight == 0.0) {
					weight = 0.001;
				}
				for (int i = 0; i < restSpaceVotes.length; i++) {
					totalVotes[i] += restSpaceVotes[i] * weight;
				}
			}
			return totalVotes;
		}
	}
}