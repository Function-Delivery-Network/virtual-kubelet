FROM golang:1.16-alpine
RUN apk --update --no-cache add tar gcc bash musl-dev ca-certificates git openssh linux-headers && rm -rf /var/cache/apk/*

ENV APISERVER_CERT_LOCATION /vkubelet-fdn-openwhisk-0-crt.pem
ENV APISERVER_KEY_LOCATION /vkubelet-fdn-openwhisk-0-key.pem
ENV KUBELET_PORT 10250
RUN apk add curl
RUN curl -sSL https://cli.openfaas.com | sh

RUN apk add curl python3 py-crcmod bash libc6-compat && rm -rf /var/cache/apk/*

RUN curl -sSL https://sdk.cloud.google.com > /tmp/gcl && bash /tmp/gcl --install-dir=/root --disable-prompts

ENV PATH="${PATH}:/root/google-cloud-sdk/bin"
RUN echo $PATH
# Use the pre-built binary in "bin/virtual-kubelet".
COPY bin/e2e/virtual-kubelet /virtual-kubelet
# Copy the configuration file for the fdn provider.
COPY ./vkubelet-fdn-openwhisk-0-cfg.json /vkubelet-fdn-openwhisk-0-cfg.json
# Copy the certificate for the HTTPS server.
COPY ./vkubelet-fdn-openwhisk-0-crt.pem /vkubelet-fdn-openwhisk-0-crt.pem
# Copy the private key for the HTTPS server.
COPY ./vkubelet-fdn-openwhisk-0-key.pem /vkubelet-fdn-openwhisk-0-key.pem

CMD ["/virtual-kubelet"]
