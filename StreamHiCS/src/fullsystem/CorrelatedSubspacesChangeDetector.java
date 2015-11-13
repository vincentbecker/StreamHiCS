package fullsystem;

import java.util.ArrayList;

import changechecker.ChangeChecker;
import changechecker.TimeCountChecker;
import contrast.Contrast;
import contrast.MicroclusterContrast;
import moa.clusterers.clustree.ClusTree;
import moa.core.ObjectRepository;
import moa.options.FloatOption;
import moa.options.IntOption;
import moa.tasks.TaskMonitor;
import subspace.Subspace;
import subspace.SubspaceSet;
import subspacebuilder.AprioriBuilder;
import subspacebuilder.SubspaceBuilder;
import weka.core.Instance;

public class CorrelatedSubspacesChangeDetector implements Callback {

	public IntOption mOption = new IntOption("m", 'm', "The number of contrast evaluations which are then averaged.",
			20, 1, Integer.MAX_VALUE);
	public FloatOption alphaOption = new FloatOption("alpha", 'a',
			"The fraction of the total weight selected for a slice.", 0.1, 0, 1);
	public FloatOption epsilonOption = new FloatOption("epsilon", 'e',
			"The deviation in contrast between subsequent evaluations of the correlated subspaces that is allowed for a subspace to stay correlated.",
			0.1, 0, 1);
	public FloatOption thresholdOption = new FloatOption("threshold", 't', "The threshold for the contrast.", 0.25, 0,
			1);
	public IntOption cutoffOption = new IntOption("cutoff", 'c',
			"The number of correlated subspaces that is used for the generation of the new candidates.", 8, 1,
			Integer.MAX_VALUE);
	public FloatOption pruningDifferenceOption = new FloatOption("pruning difference", 'p',
			"The allowed difference between the contrast between a space and a superspace for pruning.", 20, 1,
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
		streamHiCS.onAlarm();
		if (fullSpaceChangeDetector.isChangeDetected()) {
			return true;
		}
		for (SubspaceChangeDetector scd : subspaceChangeDetectors) {
			if (scd.isChangeDetected()) {
				return true;
			}
		}
		return false;
	}

	@Override
	public void onAlarm() {
		// Change subspaces and reset ChangeDetection
		fullSpaceChangeDetector.resetLearning();
		SubspaceSet correlatedSubspaces = streamHiCS.getCurrentlyCorrelatedSubspaces();
		subspaceChangeDetectors.clear();
		for (Subspace s : correlatedSubspaces.getSubspaces()) {
			SubspaceChangeDetector scd = new SubspaceChangeDetector(s);
			scd.prepareForUse();
			subspaceChangeDetectors.add(scd);
		}
	}

	public void trainOnInstanceImpl(Instance inst) {
		streamHiCS.add(inst);
		fullSpaceChangeDetector.trainOnInstance(inst);
		for (SubspaceChangeDetector scd : subspaceChangeDetectors) {
			scd.trainOnInstance(inst);
		}
	}

	public void prepareForUseImpl(TaskMonitor arg0, ObjectRepository arg1) {
		alpha = alphaOption.getValue();
		epsilon = epsilonOption.getValue();
		threshold = thresholdOption.getValue();
		cutoff = cutoffOption.getValue();
		pruningDifference = pruningDifferenceOption.getValue();

		ClusTree mcs = new ClusTree();
		mcs.resetLearningImpl();
		ChangeChecker changeChecker = new TimeCountChecker(1000);
		Contrast contrastEvaluator = new MicroclusterContrast(m, alpha, mcs);
		SubspaceBuilder subspaceBuilder = new AprioriBuilder(numberOfDimensions, threshold, cutoff, pruningDifference, contrastEvaluator);
		this.streamHiCS = new StreamHiCS(epsilon, threshold, pruningDifference, contrastEvaluator, subspaceBuilder, changeChecker, this);
		changeChecker.setCallback(streamHiCS);
		this.fullSpaceChangeDetector = new FullSpaceChangeDetector();
		fullSpaceChangeDetector.prepareForUse();
		// No correlated subspaces yet.
		this.subspaceChangeDetectors = new ArrayList<SubspaceChangeDetector>();
	}

}
