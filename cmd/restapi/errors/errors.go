package errors

type StatusError struct {
	Err  error
	Code int
}

func NewStatusError(err error, code int) StatusError {
	return StatusError{Err: err, Code: code}
}

// Error returns an associated error
func (se StatusError) Error() string {
	return se.Err.Error()
}

// Status returns an associated status code
func (se StatusError) Status() int {
	return se.Code
}
