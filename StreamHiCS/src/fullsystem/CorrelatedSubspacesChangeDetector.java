package fullsystem;

import java.util.ArrayList;
import java.util.BitSet;

import changechecker.ChangeChecker;
import changechecker.TimeCountChecker;
import clustree.ClusTree;
import environment.Stopwatch;
import moa.classifiers.AbstractClassifier;
import moa.classifiers.trees.HoeffdingTree;
import moa.core.Measurement;
import moa.core.ObjectRepository;
import moa.options.FloatOption;
import moa.options.IntOption;
import moa.tasks.TaskMonitor;
import streamdatastructures.CorrelationSummary;
import streamdatastructures.MicroclusteringAdapter;
import streamdatastructures.SummarisationAdapter;
import subspace.Subspace;
import subspace.SubspaceSet;
import subspacebuilder.AprioriBuilder;
import subspacebuilder.SubspaceBuilder;
import weka.core.Instance;

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

	/**
	 * The serial version ID.
	 */
	private static final long serialVersionUID = -686978424502737299L;

	/**
	 * The option determining the value of initial point before the correlated
	 * {@link Subspace}s are evaluated the first time.
	 */
	public IntOption initOption = new IntOption("init", 'i',
			"The number of instances after which the correlated subspaces are evaluated the first time.", 500, 1,
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
	 * Creates an instance of this class.
	 * 
	 * @param numberOfDimensions
	 *            The number of dimensions.
	 * @param streamHiCS
	 * 			The {@link StreamHiCS} instance.
	 */
	public CorrelatedSubspacesChangeDetector(int numberOfDimensions, StreamHiCS streamHiCS) {
		this.numberOfDimensions = numberOfDimensions;
		this.streamHiCS = streamHiCS;
	}

	public boolean isWarningDetected() {
		// The fullSpaceChangeDetector is only queried, if there are no subspace
		// change detectors
		if (!initialized && fullSpaceChangeDetector.isWarningDetected()) {
			return true;
		}
		for (SubspaceChangeDetector scd : subspaceChangeDetectors) {
			if (scd.isWarningDetected()) {
				return true;
			}
		}
		return false;
	}

	public boolean isChangeDetected() {
		boolean res = false;
		// The fullSpaceChangeDetector is only queried, if there are no subspace
		// change detectors
		if (!initialized && fullSpaceChangeDetector.isChangeDetected()) {
			res = true;
		}
		for (SubspaceChangeDetector scd : subspaceChangeDetectors) {
			if (scd.isChangeDetected()) {
				res = true;
				System.out.println("CHANGE detected in subspace: " + scd.getSubspace().toString());
			}
		}
		return res;
	}

	@Override
	public void onAlarm() {
		// Change subspaces and reset ChangeDetection
		// Happens anyway: fullSpaceChangeDetector.resetLearning();
		SubspaceSet correlatedSubspaces = streamHiCS.getCurrentlyCorrelatedSubspaces();
		// System.out.println("Number of samples: " + numberSamples);
		// System.out.println("Correlated: " + streamHiCS.toString());
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
		// Mark which dimensions are contained in subspaces to add detectors for all the single dimensions which are not contained in subspaces. 
		BitSet marked = new BitSet(numberOfDimensions);
		boolean found = false;
		for (Subspace s : correlatedSubspaces.getSubspaces()) {
			for(int d : s.getDimensions()){
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
		// Handle the one-dimensional subspaces
		for(int i = 0; i< numberOfDimensions; i++){
			if(!marked.get(i)){
				found = false;
				Subspace s = new Subspace(i);
				for (SubspaceChangeDetector scd : subspaceChangeDetectors) {
					// If there is an 'old' change detector running on the subspace
					// already we continue using that
					if (scd.getSubspace().size() == 1 && s.equals(scd.getSubspace())) {
						temp.add(scd);
						found = true;
					}
				}
				if (!found) {
					SubspaceChangeDetector scd = createSubspaceChangeDetector(s);
					temp.add(scd);
				}
			}
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
		}
		numberSamples++;
		if (numberSamples == initOption.getValue()) {
			// Evaluate the correlated subspaces once
			init();
		}
		for (SubspaceChangeDetector scd : subspaceChangeDetectors) {
			scd.trainOnInstance(inst);
		}
	}

	/**
	 * Calculate the correlated {@link Subspace}s and adds a change detector for
	 * each of them, if there are any
	 */
	private void init() {
		SubspaceSet correlatedSubspaces = streamHiCS.getCurrentlyCorrelatedSubspaces();
		for (Subspace s : correlatedSubspaces.getSubspaces()) {
			SubspaceChangeDetector scd = createSubspaceChangeDetector(s);
			subspaceChangeDetectors.add(scd);
		}
		initialized = true;
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
}