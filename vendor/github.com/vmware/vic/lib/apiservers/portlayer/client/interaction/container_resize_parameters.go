package interaction

// This file was generated by the swagger tool.
// Editing this file might prove futile when you re-run the swagger generate command

import (
	"net/http"
	"time"

	"golang.org/x/net/context"

	"github.com/go-openapi/errors"
	"github.com/go-openapi/runtime"
	cr "github.com/go-openapi/runtime/client"
	"github.com/go-openapi/swag"

	strfmt "github.com/go-openapi/strfmt"
)

// NewContainerResizeParams creates a new ContainerResizeParams object
// with the default values initialized.
func NewContainerResizeParams() *ContainerResizeParams {
	var ()
	return &ContainerResizeParams{

		timeout: cr.DefaultTimeout,
	}
}

// NewContainerResizeParamsWithTimeout creates a new ContainerResizeParams object
// with the default values initialized, and the ability to set a timeout on a request
func NewContainerResizeParamsWithTimeout(timeout time.Duration) *ContainerResizeParams {
	var ()
	return &ContainerResizeParams{

		timeout: timeout,
	}
}

// NewContainerResizeParamsWithContext creates a new ContainerResizeParams object
// with the default values initialized, and the ability to set a context for a request
func NewContainerResizeParamsWithContext(ctx context.Context) *ContainerResizeParams {
	var ()
	return &ContainerResizeParams{

		Context: ctx,
	}
}

// NewContainerResizeParamsWithHTTPClient creates a new ContainerResizeParams object
// with the default values initialized, and the ability to set a custom HTTPClient for a request
func NewContainerResizeParamsWithHTTPClient(client *http.Client) *ContainerResizeParams {
	var ()
	return &ContainerResizeParams{
		HTTPClient: client,
	}
}

/*ContainerResizeParams contains all the parameters to send to the API endpoint
for the container resize operation typically these are written to a http.Request
*/
type ContainerResizeParams struct {

	/*OpID*/
	OpID *string
	/*Height*/
	Height int32
	/*ID*/
	ID string
	/*Width*/
	Width int32

	timeout    time.Duration
	Context    context.Context
	HTTPClient *http.Client
}

// WithTimeout adds the timeout to the container resize params
func (o *ContainerResizeParams) WithTimeout(timeout time.Duration) *ContainerResizeParams {
	o.SetTimeout(timeout)
	return o
}

// SetTimeout adds the timeout to the container resize params
func (o *ContainerResizeParams) SetTimeout(timeout time.Duration) {
	o.timeout = timeout
}

// WithContext adds the context to the container resize params
func (o *ContainerResizeParams) WithContext(ctx context.Context) *ContainerResizeParams {
	o.SetContext(ctx)
	return o
}

// SetContext adds the context to the container resize params
func (o *ContainerResizeParams) SetContext(ctx context.Context) {
	o.Context = ctx
}

// WithHTTPClient adds the HTTPClient to the container resize params
func (o *ContainerResizeParams) WithHTTPClient(client *http.Client) *ContainerResizeParams {
	o.SetHTTPClient(client)
	return o
}

// SetHTTPClient adds the HTTPClient to the container resize params
func (o *ContainerResizeParams) SetHTTPClient(client *http.Client) {
	o.HTTPClient = client
}

// WithOpID adds the opID to the container resize params
func (o *ContainerResizeParams) WithOpID(opID *string) *ContainerResizeParams {
	o.SetOpID(opID)
	return o
}

// SetOpID adds the opId to the container resize params
func (o *ContainerResizeParams) SetOpID(opID *string) {
	o.OpID = opID
}

// WithHeight adds the height to the container resize params
func (o *ContainerResizeParams) WithHeight(height int32) *ContainerResizeParams {
	o.SetHeight(height)
	return o
}

// SetHeight adds the height to the container resize params
func (o *ContainerResizeParams) SetHeight(height int32) {
	o.Height = height
}

// WithID adds the id to the container resize params
func (o *ContainerResizeParams) WithID(id string) *ContainerResizeParams {
	o.SetID(id)
	return o
}

// SetID adds the id to the container resize params
func (o *ContainerResizeParams) SetID(id string) {
	o.ID = id
}

// WithWidth adds the width to the container resize params
func (o *ContainerResizeParams) WithWidth(width int32) *ContainerResizeParams {
	o.SetWidth(width)
	return o
}

// SetWidth adds the width to the container resize params
func (o *ContainerResizeParams) SetWidth(width int32) {
	o.Width = width
}

// WriteToRequest writes these params to a swagger request
func (o *ContainerResizeParams) WriteToRequest(r runtime.ClientRequest, reg strfmt.Registry) error {

	r.SetTimeout(o.timeout)
	var res []error

	if o.OpID != nil {

		// header param Op-ID
		if err := r.SetHeaderParam("Op-ID", *o.OpID); err != nil {
			return err
		}

	}

	// query param height
	qrHeight := o.Height
	qHeight := swag.FormatInt32(qrHeight)
	if qHeight != "" {
		if err := r.SetQueryParam("height", qHeight); err != nil {
			return err
		}
	}

	// path param id
	if err := r.SetPathParam("id", o.ID); err != nil {
		return err
	}

	// query param width
	qrWidth := o.Width
	qWidth := swag.FormatInt32(qrWidth)
	if qWidth != "" {
		if err := r.SetQueryParam("width", qWidth); err != nil {
			return err
		}
	}

	if len(res) > 0 {
		return errors.CompositeValidationError(res...)
	}
	return nil
}
