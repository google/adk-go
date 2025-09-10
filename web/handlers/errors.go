package handlers

import (
	"net/http"
)

type StatusError struct {
	error
	Code int
}

// Error returns an associated error
func (se StatusError) Unwrap() error {
	return se.error
}

// Status returns an associated status code
func (se StatusError) Status() int {
	return se.Code
}

type ErrorHandler func(http.ResponseWriter, *http.Request) error

func FromErrorHandler(fn ErrorHandler) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		err := fn(w, r)
		if err != nil {
			if statusErr, ok := err.(StatusError); ok {
				http.Error(w, statusErr.Error(), statusErr.Status())
			} else {
				http.Error(w, err.Error(), http.StatusInternalServerError)
			}
		}
	}
}
