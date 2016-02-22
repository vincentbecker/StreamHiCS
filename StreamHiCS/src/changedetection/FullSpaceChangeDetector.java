package changedetection;

public class FullSpaceChangeDetector extends SingleClassifierDrift implements ChangeDetector{

	/**
	 * The serial version ID. 
	 */
	private static final long serialVersionUID = 6901371320471962273L;

	@Override
	public boolean isWarningDetected() {
		return (this.ddmLevel == DDM_WARNING_LEVEL);
	}
	 
	@Override
	public boolean isChangeDetected() {
	    return (this.ddmLevel == DDM_OUTCONTROL_LEVEL);
	}
	
}
