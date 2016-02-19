package changedetection;

public class FullSpaceChangeDetector extends SingleClassifierDrift implements ChangeDetector{

	/**
	 * The serial version ID. 
	 */
	private static final long serialVersionUID = 6901371320471962273L;

	public boolean isWarningDetected() {
		return (this.ddmLevel == DDM_WARNING_LEVEL);
	}
	 
	public boolean isChangeDetected() {
	    return (this.ddmLevel == DDM_OUTCONTROL_LEVEL);
	}
	
}