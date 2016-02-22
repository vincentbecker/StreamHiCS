package changedetection;

import java.util.ArrayList;
import java.util.BitSet;

import changedetection.SubspaceChangeDetectors.State;
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
public class SubspaceClassifiersChangeDetector extends AbstractClassifier implements Callback, ChangeDetector {

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

	public FlagOption useRestspaceOption = new FlagOption("useRestspace", 'u',
			"Whether the restspace should be included in the change detection process.");

	/**
	 * The {@link StreamHiCS} instance.
	 */
	private StreamHiCS streamHiCS;

	/**
	 * The {@link FullSpaceChangeDetector} instance.
	 */
	private HoeffdingTree fullSpaceClassifier;

	private HoeffdingTree newFullSpaceClassifier;

	/**
	 * The {@link ArrayList} containing the {@link SubspaceChangeDetector}s,
	 * i.e. the change detector running on each correlated {@link Subspace}.
	 */
	private ArrayList<SubspaceClassifier> subspaceClassifiers;

	private ArrayList<SubspaceClassifier> newSubspaceClassifiers;

	private boolean newClassifiersReset;

	/**
	 * A {@link SubspaceChangeDetector} running on all dimensions which are not
	 * contained in the correlated subspaces.
	 */
	private RestspaceClassifier restSpaceClassifier;

	private RestspaceClassifier newRestSpaceClassifier;

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
	 * The {@link DDM} instance.
	 */
	private DDM ddm;

	/**
	 * Creates an instance of this class.
	 * 
	 * @param numberOfDimensions
	 *            The number of dimensions.
	 * @param streamHiCS
	 *            The {@link StreamHiCS} instance.
	 */
	public SubspaceClassifiersChangeDetector(int numberOfDimensions, StreamHiCS streamHiCS) {
		this.numberOfDimensions = numberOfDimensions;
		this.streamHiCS = streamHiCS;
	}

	/**
	 * Checks for the warning state.
	 * 
	 * @return True, if in warning state, false otherwise.
	 */
	public boolean isWarningDetected() {
		return (this.state == State.WARNING);
	}

	/**
	 * Checks for the drift state.
	 * 
	 * @return True, if in drift state, false otherwise.
	 */
	public boolean isChangeDetected() {
		return (this.state == State.DRIFT);
	}

	@Override
	public void onAlarm() {
		// Change subspaces and reset classifiers
		if (initialized) {
			SubspaceSet correlatedSubspaces = streamHiCS.getCurrentlyCorrelatedSubspaces();
			 System.out.println("SCCD: Correlated: " + correlatedSubspaces.toString() + " at " + numberSamples);
			if (createSubspaceClassifiers(correlatedSubspaces)) {
				ddm.resetLearning();
			}
		}
	}

	@Override
	public void trainOnInstanceImpl(Instance inst) {
		// streamHiCS.add(inst);
		if (numberSamples == numberInit) {
			// Evaluate the correlated subspaces once
			init();
		}
		int trueClass = (int) inst.classValue();
		double prediction;
		if (getClassPrediction(inst) == trueClass) {
			prediction = 0.0;
		} else {
			prediction = 1.0;
		}
		this.ddm.input(prediction);
		if (ddm.getChange()) {
			state = State.DRIFT;
		} else if (ddm.getWarningZone()) {
			state = State.WARNING;
		} else {
			state = State.IN_CONTROL;
		}
		handleState();

		if (!initialized) {
			fullSpaceClassifier.trainOnInstance(inst);
		} else {
			for (SubspaceClassifier sc : subspaceClassifiers) {
				sc.trainOnInstance(inst);
			}
			if (restSpaceClassifier != null && (subspaceClassifiers.isEmpty() || useRestspace)) {
				restSpaceClassifier.trainOnInstance(inst);
			}
		}
		if (isWarningDetected()) {
			// Train new classifiers, too
			if (!initialized) {
				newFullSpaceClassifier.trainOnInstance(inst);
			} else {
				for (SubspaceClassifier sc : newSubspaceClassifiers) {
					sc.trainOnInstance(inst);
				}
				if (newRestSpaceClassifier != null && (newSubspaceClassifiers.isEmpty() || useRestspace)) {
					newRestSpaceClassifier.trainOnInstance(inst);
				}
			}
		}
		numberSamples++;
	}

	/**
	 * The reaction to a state. 
	 */
	private void handleState() {
		switch (this.state) {
		case WARNING:
			if (newClassifiersReset) {
				if (!initialized) {
					newFullSpaceClassifier.resetLearning();
				} else {
					for (SubspaceClassifier sc : newSubspaceClassifiers) {
						sc.resetLearning();
					}
					if (restSpaceClassifier != null) {
						newRestSpaceClassifier.resetLearning();
					}
				}
				newClassifiersReset = false;
			}
			break;
		case DRIFT:
			if (!initialized) {
				fullSpaceClassifier = newFullSpaceClassifier;
				newFullSpaceClassifier = new HoeffdingTree();
				newFullSpaceClassifier.prepareForUse();
			} else {
				Subspace s;
				for (int i = 0; i < subspaceClassifiers.size(); i++) {
					s = subspaceClassifiers.get(i).getSubspace();
					subspaceClassifiers.set(i, newSubspaceClassifiers.get(i));
					newSubspaceClassifiers.set(i, createSubspaceClassifier(s));
				}
				if (restSpaceClassifier != null) {
					s = restSpaceClassifier.getSubspace();
					restSpaceClassifier = newRestSpaceClassifier;
					newRestSpaceClassifier = createRestspaceClassifier(s);
				}
			}
			break;
		case IN_CONTROL:
			newClassifiersReset = true;
			break;
		default:
		}
	}

	/**
	 * Creates and initialises a {@link SubspaceClassifier}s for a given
	 * {@link SubspaceSet}, including the classifier for the restspace.
	 * 
	 * @param s
	 *            The {@link SubspaceSet}
	 * @return True, if anything has changed since the last call of this method, false otherwise. 
	 */
	private boolean createSubspaceClassifiers(SubspaceSet subspaceSet) {
		ArrayList<SubspaceClassifier> temp = new ArrayList<SubspaceClassifier>();
		ArrayList<SubspaceClassifier> newTemp = new ArrayList<SubspaceClassifier>();
		// Mark which dimensions are contained in subspaces to add detectors for
		// all the single dimensions which are not contained in subspaces.
		BitSet marked = new BitSet(numberOfDimensions);
		boolean found = false;
		boolean update = false;
		for (Subspace s : subspaceSet.getSubspaces()) {
			// s.sort();
			for (int d : s.getDimensions()) {
				marked.set(d);
			}
			found = false;
			SubspaceClassifier sc;
			for (int i = 0; i < subspaceClassifiers.size() && !found; i++) {
				sc = subspaceClassifiers.get(i);
				// If there is an 'old' change detector running on the subspace
				// already we continue using that
				if (s.equals(sc.getSubspace())) {
					temp.add(sc);
					newTemp.add(newSubspaceClassifiers.get(i));
					found = true;
				}
			}
			// If the subspace is new we start a new change detector on it
			if (!found) {
				update = true;
				temp.add(createSubspaceClassifier(s));
				newTemp.add(createSubspaceClassifier(s));
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
			if ((useRestspace || subspaceSet.isEmpty())
					&& (restSpaceClassifier == null || !restSpace.equals(restSpaceClassifier.getSubspace()))) {
				restSpaceClassifier = createRestspaceClassifier(restSpace);
				newRestSpaceClassifier = createRestspaceClassifier(restSpace);
				update = true;
			}
		} else {
			restSpaceClassifier = null;
			newRestSpaceClassifier = null;
		}
		if (subspaceClassifiers.size() != temp.size()) {
			update = true;
		}
		subspaceClassifiers = temp;
		newSubspaceClassifiers = newTemp;
		return update;
	}

	/**
	 * Initialise the {@link SubspaceClassifier}s.
	 */
	private void init() {
		initialized = true;
		if (subspaceClassifiers.isEmpty() && restSpaceClassifier == null) {
			onAlarm();
		}
		// Reset the DDM instance?
		ddm.resetLearning();
	}

	/**
	 * Creates and initialises a {@link SubspaceClassifier} for a given
	 * {@link Subspace}.
	 * 
	 * @param s
	 *            The subspace
	 * @return The {@link SubspaceClassifier}.
	 */
	private SubspaceClassifier createSubspaceClassifier(Subspace s) {
		SubspaceClassifier sc = new SubspaceClassifier(s);
		sc.prepareForUse();
		return sc;
	}

	/**
	 * The same procedure as above for creating a {@link SubspaceClassifier}
	 * for the restspace. The only difference is that {@link NaiveBayes} instead
	 * of the {@link HoeffdingTree} is used.
	 * 
	 * @param restspace The {@link Subspace} representing the restspace
	 * @return A {@link SubspaceClsasifier} for the restspace. 
	 */
	private RestspaceClassifier createRestspaceClassifier(Subspace restspace) {
		RestspaceClassifier restSpaceClassifier = new RestspaceClassifier(restspace);
		restSpaceClassifier.prepareForUse();
		return restSpaceClassifier;
	}

	@Override
	public void prepareForUseImpl(TaskMonitor arg0, ObjectRepository arg1) {
		useRestspace = useRestspaceOption.isSet();
		// ChangeDetectors
		this.fullSpaceClassifier = new HoeffdingTree();
		fullSpaceClassifier.prepareForUse();
		this.newFullSpaceClassifier = new HoeffdingTree();
		newFullSpaceClassifier.prepareForUse();
		// No correlated subspaces yet.
		this.subspaceClassifiers = new ArrayList<SubspaceClassifier>();
		this.newSubspaceClassifiers = new ArrayList<SubspaceClassifier>();
		this.numberInit = initOption.getValue();
		this.ddm = new DDM();
		this.ddm.prepareForUse();
		this.state = State.IN_CONTROL;
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
		fullSpaceClassifier.resetLearning();
		newFullSpaceClassifier.resetLearning();
		numberSamples = 0;
		subspaceClassifiers.clear();
		newSubspaceClassifiers.clear();
		restSpaceClassifier = null;
		newRestSpaceClassifier = null;
		streamHiCS.clear();
		initialized = false;
		this.state = State.IN_CONTROL;
		this.ddm.resetLearning();
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
			return fullSpaceClassifier.getVotesForInstance(instance);
		} else {
			int numberClasses = instance.classAttribute().numValues();
			double[] totalVotes = new double[numberClasses];

			double weight = 0;
			for (SubspaceClassifier sc : subspaceClassifiers) {
				double[] scVotes = sc.getVotesForInstance(instance);
				weight = sc.getAccuracy();
				if (weight == 0.0) {
					weight = 0.001;
				}
				for (int i = 0; i < scVotes.length; i++) {
					totalVotes[i] += scVotes[i] * weight;
				}
			}

			if (restSpaceClassifier != null && (subspaceClassifiers.isEmpty() || useRestspace)) {
				double[] restSpaceVotes = restSpaceClassifier.getVotesForInstance(instance);
				// weight = restSpaceClassifier.getSubspace().size();
				weight = restSpaceClassifier.getAccuracy();
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