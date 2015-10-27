package changechecker;

/**
 * A default checking mechanism which returns true every time it is called and
 * thereby induces a new evaluation of the correlated subspaces.
 * 
 * @author Vincent
 *
 */
public class TimeCountChecker extends ChangeChecker {
	
	public TimeCountChecker(int checkInterval) {
		super(checkInterval);
	}

	@Override
	public boolean checkForChange() {
		return true;
	}

}
