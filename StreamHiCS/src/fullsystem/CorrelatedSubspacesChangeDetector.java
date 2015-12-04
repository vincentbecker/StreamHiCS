package fullsystem;

import java.util.ArrayList;

import changechecker.ChangeChecker;
import changechecker.TimeCountChecker;
import clustree.ClusTree;
import contrast.Contrast;
import contrast.MicroclusterContrast;
import moa.classifiers.AbstractClassifier;
//import moa.clusterers.clustree.ClusTree;
import moa.core.Measurement;
import moa.core.ObjectRepository;
import moa.options.FloatOption;
import moa.options.IntOption;
import moa.tasks.TaskMonitor;
import subspace.Subspace;
import subspace.SubspaceSet;
import subspacebuilder.AprioriBuilder;
import subspacebuilder.SubspaceBuilder;
import weka.core.Instance;

public class CorrelatedSubspacesChangeDetector extends AbstractClassifier implements Callback {

	/**
	 * 
	 */
	private static final long serialVersionUID = -686978424502737299L;

	public IntOption mOption = new IntOption("m", 'm', "The number of contrast evaluations which are then averaged.",
			50, 1, Integer.MAX_VALUE);
	public FloatOption alphaOption = new FloatOption("alpha", 'a',
			"The fraction of the total weight selected for a slice.", 0.1, 0, 1);
	public FloatOption epsilonOption = new FloatOption("epsilon", 'e',
			"The deviation in contrast between subsequent evaluations of the correlated subspaces that is allowed for a subspace to stay correlated.",
			0.1, 0, 1);
	public FloatOption thresholdOption = new FloatOption("threshold", 't', "The threshold for the contrast.", 0.5, 0,
			1);
	public IntOption cutoffOption = new IntOption("cutoff", 'c',
			"The number of correlated subspaces that is used for the generation of the new candidates.", 8, 1,
			Integer.MAX_VALUE);
	public FloatOption pruningDifferenceOption = new FloatOption("pruningDifference", 'p',
			"The allowed difference between the contrast between a space and a superspace for pruning.", 0.1, 0, 1);
	public IntOption initOption = new IntOption("init", 'i',
			"The number of instances after which the correlated subspaces are evaluated the first time.", 500, 1,
			Integer.MAX_VALUE);

	private StreamHiCS streamHiCS;
	private FullSpaceChangeDetector fullSpaceChangeDetector;
	private ArrayList<SubspaceChangeDetector> subspaceChangeDetectors;
	private int numberOfDimensions;
	private int m;
	private double alpha;
	private double epsilon;
	private double threshold;
	private int cutoff;
	private double pruningDifference;
	private int numberSamples;

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
		System.out.println("Number of samples: " + numberSamples);
		System.out.println("Number of microclusters: " + streamHiCS.getNumberOfElements());
		System.out.println("Correlated: " + streamHiCS.toString());
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
				SubspaceChangeDetector scd = new SubspaceChangeDetector(s);
				scd.prepareForUse();
				temp.add(scd);
			}
		}
		subspaceChangeDetectors = temp;
	}

	@Override
	public void trainOnInstanceImpl(Instance inst) {
		streamHiCS.add(inst);
		if(inst.classValue() < 0){
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

	private void init() {
		SubspaceSet correlatedSubspaces = streamHiCS.getCurrentlyCorrelatedSubspaces();
		for (Subspace s : correlatedSubspaces.getSubspaces()) {
			SubspaceChangeDetector scd = new SubspaceChangeDetector(s);
			scd.prepareForUse();
			subspaceChangeDetectors.add(scd);
		}
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
		mcs.horizonOption.setValue(1000);
		mcs.resetLearningImpl();
		ChangeChecker changeChecker = new TimeCountChecker(1000);
		Contrast contrastEvaluator = new MicroclusterContrast(m, alpha, mcs);
		SubspaceBuilder subspaceBuilder = new AprioriBuilder(numberOfDimensions, threshold, cutoff, pruningDifference,
				contrastEvaluator);
		streamHiCS = new StreamHiCS(epsilon, threshold, pruningDifference, contrastEvaluator, subspaceBuilder,
				changeChecker, this);
		changeChecker.setCallback(streamHiCS);
		this.fullSpaceChangeDetector = new FullSpaceChangeDetector();
		fullSpaceChangeDetector.prepareForUse();
		// No correlated subspaces yet.
		this.subspaceChangeDetectors = new ArrayList<SubspaceChangeDetector>();
		super.prepareForUseImpl(arg0, arg1);
	}

	public int getNumberOfElements() {
		return streamHiCS.getNumberOfElements();
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
