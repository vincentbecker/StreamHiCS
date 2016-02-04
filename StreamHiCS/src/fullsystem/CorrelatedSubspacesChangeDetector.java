package fullsystem;

import java.util.ArrayList;
import java.util.BitSet;

import moa.classifiers.AbstractClassifier;
import moa.classifiers.trees.HoeffdingTree;
import moa.core.Measurement;
import moa.core.ObjectRepository;
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
public class CorrelatedSubspacesChangeDetector extends AbstractClassifier implements Callback, ChangeDetector {

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

	private State state;

	/**
	 * Creates an instance of this class.
	 * 
	 * @param numberOfDimensions
	 *            The number of dimensions.
	 * @param streamHiCS
	 *            The {@link StreamHiCS} instance.
	 */
	public CorrelatedSubspacesChangeDetector(int numberOfDimensions, StreamHiCS streamHiCS) {
		this.numberOfDimensions = numberOfDimensions;
		this.streamHiCS = streamHiCS;
	}

	public boolean isWarningDetected() {
		if (state == State.WARNING) {
			return true;
		}
		return false;
	}

	public boolean isChangeDetected() {
		if (state == State.DRIFT) {
			return true;
		}

		return false;
	}

	@Override
	public void onAlarm() {
		// Change subspaces and reset ChangeDetection
		SubspaceSet correlatedSubspaces = streamHiCS.getCurrentlyCorrelatedSubspaces();
		// System.out.println("Number of samples: " + numberSamples);
		/*
		SubspaceSet correlatedSubspaces = new SubspaceSet();
		Subspace cs = new Subspace(0, 2, 3, 7, 8);
		correlatedSubspaces.addSubspace(cs);
		*/
		/*
		 * SubspaceSet correlatedSubspaces = new SubspaceSet(); if
		 * (numberSamples < 0) { Subspace s1 = new Subspace(3, 4, 7); Subspace
		 * s2 = new Subspace(5, 6, 7); correlatedSubspaces.addSubspace(s1);
		 * correlatedSubspaces.addSubspace(s2); } else { Subspace s1 = new
		 * Subspace(0, 1, 2, 3, 4, 5, 6); correlatedSubspaces.addSubspace(s1); }
		 */

		System.out.println("Correlated: " + correlatedSubspaces.toString() + " at " + numberSamples);
		// subspaceChangeDetectors.clear();
		/*
		 * System.out.println("Number of samples: " + numberSamples);
		 * System.out.println("Number of microclusters: " +
		 * streamHiCS.getNumberOfElements()); System.out.println("Correlated: "
		 * + streamHiCS.toString()); for (Subspace s :
		 * correlatedSubspaces.getSubspaces()) {
		 * System.out.print(s.getContrast() + ", "); } System.out.println();
		 */
		ArrayList<SubspaceChangeDetector> temp = new ArrayList<SubspaceChangeDetector>();
		// Mark which dimensions are contained in subspaces to add detectors for
		// all the single dimensions which are not contained in subspaces.
		BitSet marked = new BitSet(numberOfDimensions);
		boolean found = false;
		for (Subspace s : correlatedSubspaces.getSubspaces()) {
			for (int d : s.getDimensions()) {
				marked.set(d);
			}
			found = false;
			for (SubspaceChangeDetector scd : subspaceChangeDetectors) {
				// If there is an 'old' change detector running on the subspace
				// already we continue using that
				if (s.equals(scd.getSubspace())) {
					temp.add(scd);
					found = true;
				}
			}
			// If the subspace is new we start a new change detector on it
			if (!found) {
				SubspaceChangeDetector scd = createSubspaceChangeDetector(s);
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
			SubspaceChangeDetector restSpaceChangeDetector = createRestspaceDetector(restSpace);
			temp.add(restSpaceChangeDetector);
		}
		subspaceChangeDetectors = temp;
		initialized = true;
	}

	@Override
	public void trainOnInstanceImpl(Instance inst) {
		streamHiCS.add(inst);
		if (inst.classValue() < 0) {
			System.out.println("Class value < 0");
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
		}
		numberSamples++;
		if (numberSamples == numberInit) {
			// Evaluate the correlated subspaces once
			init();
		}
		boolean warning = false;
		boolean drift = false;
		ArrayList<Integer> driftDetectedIndexes = new ArrayList<Integer>();
		for (int i = 0; i < subspaceChangeDetectors.size(); i++) {
			SubspaceChangeDetector scd = subspaceChangeDetectors.get(i);
			scd.trainOnInstance(inst);
			if (scd.isWarningDetected()) {
				warning = true;
			} else if (scd.isChangeDetected()) {
				if(streamHiCS.isValidSubspace(scd.getSubspace())){
					System.out.println("Change in subspace: " + scd.getSubspace().toString());
					drift = true;
					driftDetectedIndexes.add(i);
				}else{
					System.out.println("Would have been change in subspace: " + scd.getSubspace().toString());
					streamHiCS.evaluateCorrelatedSubspaces();
				}
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

	private void driftOccurred(ArrayList<Integer> driftDetectedIndexes) {
		for (int i = 0; i < subspaceChangeDetectors.size(); i++) {
			if (!driftDetectedIndexes.contains(i)) {
				subspaceChangeDetectors.get(i).resetLearning();
			}
		}
	}

	/**
	 * Calculate the correlated {@link Subspace}s and adds a change detector for
	 * each of them, if there are any
	 */
	private void init() {
		streamHiCS.onAlarm();
		// SubspaceSet correlatedSubspaces =
		// streamHiCS.getCurrentlyCorrelatedSubspaces();
		if (subspaceChangeDetectors.isEmpty()) {
			Subspace restspace = new Subspace();
			for (int i = 0; i < numberOfDimensions; i++) {
				restspace.addDimension(i);
			}
			SubspaceChangeDetector restSpaceChangeDetector = createRestspaceDetector(restspace);
			subspaceChangeDetectors.add(restSpaceChangeDetector);
		}
		initialized = true;
	}

	private SubspaceChangeDetector createRestspaceDetector(Subspace restspace) {
		SubspaceChangeDetector restSpaceChangeDetector = new SubspaceChangeDetector(restspace);
		AbstractClassifier baseLearner = new HoeffdingTree();
		baseLearner.prepareForUse();
		restSpaceChangeDetector.baseLearnerOption.setCurrentObject(baseLearner);
		restSpaceChangeDetector.prepareForUse();
		return restSpaceChangeDetector;
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
		// AbstractClassifier baseLearner = new HoeffdingAdaptiveTree();
		// AbstractClassifier baseLearner = new DecisionStump();
		baseLearner.prepareForUse();
		scd.baseLearnerOption.setCurrentObject(baseLearner);
		scd.prepareForUse();

		return scd;
	}

	@Override
	public void prepareForUseImpl(TaskMonitor arg0, ObjectRepository arg1) {
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
		streamHiCS.clear();
		initialized = false;
	}

	@Override
	public double[] getVotesForInstance(Instance arg0) {
		// TODO Auto-generated method stub
		return null;
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
		if (numberSamples <= numberInit) {
			return Utils.maxIndex(fullSpaceChangeDetector.getVotesForInstance(instance));
		} else {
			int numberClasses = instance.classAttribute().numValues();
			double[] totalPredictions = new double[numberClasses];

			for (SubspaceChangeDetector scd : subspaceChangeDetectors) {
				double[] scdPrediction = scd.getVotesForInstance(instance);
				if (scdPrediction.length == numberClasses) {
					for (int i = 0; i < numberClasses; i++) {
						totalPredictions[i] += scdPrediction[i];
					}
				}
			}

			//totalPredictions = subspaceChangeDetectors.get(0).getVotesForInstance(instance);
			return Utils.maxIndex(totalPredictions);
		}
	}
}