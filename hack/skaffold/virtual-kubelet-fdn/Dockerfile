FROM golang:1.16-alpine
RUN apk --update --no-cache add tar gcc bash musl-dev ca-certificates git openssh linux-headers && rm -rf /var/cache/apk/*

ENV APISERVER_CERT_LOCATION /vkubelet-fdn-openwhisk-0-crt.pem
ENV APISERVER_KEY_LOCATION /vkubelet-fdn-openwhisk-0-key.pem
ENV KUBELET_PORT 10250
RUN apk add curl
RUN curl -sSL https://cli.openfaas.com | sh
# Use the pre-built binary in "bin/virtual-kubelet".
COPY bin/e2e/virtual-kubelet /virtual-kubelet
# Copy the configuration file for the fdn provider.
COPY ./vkubelet-fdn-openwhisk-0-cfg.json /vkubelet-fdn-openwhisk-0-cfg.json
# Copy the certificate for the HTTPS server.
COPY ./vkubelet-fdn-openwhisk-0-crt.pem /vkubelet-fdn-openwhisk-0-crt.pem
# Copy the private key for the HTTPS server.
COPY ./vkubelet-fdn-openwhisk-0-key.pem /vkubelet-fdn-openwhisk-0-key.pem

CMD ["/virtual-kubelet"]
