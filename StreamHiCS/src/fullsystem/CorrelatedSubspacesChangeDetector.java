package fullsystem;

import java.util.ArrayList;

import changechecker.ChangeChecker;
import changechecker.TimeCountChecker;
import clustree.ClusTree;
import environment.Stopwatch;
import moa.classifiers.AbstractClassifier;
import moa.classifiers.trees.HoeffdingAdaptiveTree;
import moa.core.Measurement;
import moa.core.ObjectRepository;
import moa.options.FloatOption;
import moa.options.IntOption;
import moa.tasks.TaskMonitor;
import streamdatastructures.MicroclusterAdapter;
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
	 * The option determining the value of m, i.e. how many iterations are
	 * carried out to calculate the contrast of a {@link Subspace}.
	 */
	public IntOption mOption = new IntOption("m", 'm', "The number of contrast evaluations which are then averaged.",
			50, 1, Integer.MAX_VALUE);

	/**
	 * The option determining the value of alpha, i.e. how big the slice is as a
	 * fraction of the full sample.
	 */
	public FloatOption alphaOption = new FloatOption("alpha", 'a',
			"The fraction of the total weight selected for a slice.", 0.1, 0, 1);

	/**
	 * The option determining the value of epsilon, i.e. much the contrast of a
	 * {@link Subspace} may change without signalling a change.
	 */
	public FloatOption epsilonOption = new FloatOption("epsilon", 'e',
			"The deviation in contrast between subsequent evaluations of the correlated subspaces that is allowed for a subspace to stay correlated.",
			0.1, 0, 1);

	/**
	 * The option determining the value of the threshold, i.e. how high the
	 * contrast of a {@link Subspace} must be so that it is considered as
	 * correlated. }.
	 */
	public FloatOption thresholdOption = new FloatOption("threshold", 't', "The threshold for the contrast.", 0.5, 0,
			1);

	/**
	 * The option determining the value of the cutoff, i.e. how many solutions
	 * are allowed per step in the buildup process of the correlated
	 * {@link Subspace}.
	 */
	public IntOption cutoffOption = new IntOption("cutoff", 'c',
			"The number of correlated subspaces that is used for the generation of the new candidates.", 8, 1,
			Integer.MAX_VALUE);

	/**
	 * The option determining the value of the pruning difference, i.e. how
	 * large the difference in contrast may be so that a subspace is pruned, is
	 * a correlated superspace exists.
	 */
	public FloatOption pruningDifferenceOption = new FloatOption("pruningDifference", 'p',
			"The allowed difference between the contrast between a space and a superspace for pruning.", 0.1, 0, 1);

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
	 * The number of how many iterations are carried out to calculate the
	 * contrast of a {@link Subspace}.
	 */
	private int m;

	/**
	 * The size of the slice as a fraction of the size of the full sample.
	 */
	private double alpha;

	/**
	 * How much the contrast of a {@link Subspace} may change without signalling
	 * a change.
	 */
	private double epsilon;

	/**
	 * How high the contrast of a {@link Subspace} must be so that it is
	 * considered as correlated. }.
	 */
	private double threshold;

	/**
	 * How many solutions are allowed per step in the buildup process of the
	 * correlated {@link Subspace}.
	 */
	private int cutoff;

	/**
	 * How large the difference in contrast may be so that a subspace is pruned,
	 * is a correlated superspace exists.
	 */
	private double pruningDifference;

	/**
	 * The number of samples seen.
	 */
	private int numberSamples;

	/**
	 * Creates an instance of this class.
	 * 
	 * @param numberOfDimensions
	 *            The number of dimensions.
	 */
	public CorrelatedSubspacesChangeDetector(int numberOfDimensions) {
		this.numberOfDimensions = numberOfDimensions;
	}

	public boolean isWarningDetected() {
		if (fullSpaceChangeDetector.isWarningDetected()) {
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
		if (fullSpaceChangeDetector.isChangeDetected()) {
			res = true;
		}
		for (SubspaceChangeDetector scd : subspaceChangeDetectors) {
			if (scd.isChangeDetected()) {
				res = true;
			}
		}

		if (res) {
			streamHiCS.onAlarm();
		}
		return res;
	}

	@Override
	public void onAlarm() {
		// Change subspaces and reset ChangeDetection
		// Happens anyway: fullSpaceChangeDetector.resetLearning();
		SubspaceSet correlatedSubspaces = streamHiCS.getCurrentlyCorrelatedSubspaces();
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
		boolean found = false;
		for (Subspace s : correlatedSubspaces.getSubspaces()) {
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
		subspaceChangeDetectors = temp;
	}

	@Override
	public void trainOnInstanceImpl(Instance inst) {
		streamHiCS.add(inst);
		if (inst.classValue() < 0) {
			System.out.println("Class value < 0");
		}
		fullSpaceChangeDetector.trainOnInstance(inst);
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
		AbstractClassifier baseLearner = new HoeffdingAdaptiveTree();
		// AbstractClassifier baseLearner = new DecisionStump();
		baseLearner.prepareForUse();
		scd.baseLearnerOption.setCurrentObject(baseLearner);
		scd.prepareForUse();

		return scd;
	}

	@Override
	public void prepareForUseImpl(TaskMonitor arg0, ObjectRepository arg1) {
		m = mOption.getValue();
		alpha = alphaOption.getValue();
		epsilon = epsilonOption.getValue();
		threshold = thresholdOption.getValue();
		cutoff = cutoffOption.getValue();
		pruningDifference = pruningDifferenceOption.getValue();

		ClusTree mcs = new ClusTree();
		mcs.horizonOption.setValue(2000);
		// mcs.horizonOption.setValue(20000);
		mcs.prepareForUse();

		/*
		 * WithDBSCAN mcs = new WithDBSCAN(); mcs.speedOption.setValue(100);
		 * mcs.epsilonOption.setValue(0.4); mcs.betaOption.setValue(0.1);
		 * mcs.lambdaOption.setValue(0.05); mcs.prepareForUse();
		 */
		// StreamHiCS
		SummarisationAdapter adapter = new MicroclusterAdapter(mcs);
		Contrast contrastEvaluator = new Contrast(m, alpha, adapter);
		ChangeChecker changeChecker = new TimeCountChecker(1000);
		SubspaceBuilder subspaceBuilder = new AprioriBuilder(numberOfDimensions, threshold, cutoff, pruningDifference,
				contrastEvaluator);
		Stopwatch stopwatch = new Stopwatch();
		streamHiCS = new StreamHiCS(epsilon, threshold, pruningDifference, contrastEvaluator, subspaceBuilder,
				changeChecker, this, stopwatch);
		changeChecker.setCallback(streamHiCS);

		// ChangeDetectors
		this.fullSpaceChangeDetector = new FullSpaceChangeDetector();
		AbstractClassifier baseLearner = new HoeffdingAdaptiveTree();
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
	 * Returns the currently correlated {@link Subspace}s in a {@link SubspaceSet}. 
	 * 
	 * @return The currently correlated {@link Subspace}s in a {@link SubspaceSet}. 
	 */
	public SubspaceSet getCurrentlyCorrelatedSubspaces() {
		return streamHiCS.getCurrentlyCorrelatedSubspaces();
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

	@Override
	public void resetLearningImpl() {
		fullSpaceChangeDetector.resetLearning();
		numberSamples = 0;
		subspaceChangeDetectors.clear();
		streamHiCS.clear();
	}

}